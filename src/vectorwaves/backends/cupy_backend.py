import numpy as np
try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False

class CupyMethods:
    def __init__(self, beam, use_single_precision=True):
        if not has_cupy:
            raise RuntimeError("CuPy/CUDA not found.")
            
        self.use_single = use_single_precision
        self.real_dt = cp.float32   if self.use_single else cp.float64
        self.comp_dt = cp.complex64 if self.use_single else cp.complex128

        # --- Persistent Model Data ---
        def to_gpu(arr, dtype): return cp.ascontiguousarray(cp.asarray(arr, dtype=dtype))

        self.kx = to_gpu(beam.k[0], self.real_dt)
        self.ky = to_gpu(beam.k[1], self.real_dt)
        self.kz = to_gpu(beam.k[2], self.real_dt)
        self.w  = to_gpu(beam.w, self.real_dt)
        self.inv_w = to_gpu(beam.inv_w, self.real_dt)
        self.c_base = to_gpu(beam.c, self.comp_dt)
        self.num_waves = len(self.w)
        
        self._kernel_cache = {}

    def _get_kernel(self, num_components, is_grid=False):
        key = (num_components, is_grid)
        if key in self._kernel_cache: return self._kernel_cache[key]

        real_t = "float" if self.use_single else "double"
        comp_t = "complex<float>" if self.use_single else "complex<double>"
        sincos_f = "sincosf" if self.use_single else "sincos"

        coord_logic = """
            int ix = p % nx;
            int iy = p / nx;
            {real_t} px = x_vec[ix];
            {real_t} py = y_vec[iy];
            {real_t} pz = z_scalar;
        """ if is_grid else """
            {real_t} px = x_vec[p];
            {real_t} py = y_vec[p];
            {real_t} pz = z_vec[p];
        """

        kernel_code = f"""
        #include <cupy/complex.cuh>
        extern "C" __global__
        void compute_kernel(
            const {real_t}* __restrict__ x_vec, const {real_t}* __restrict__ y_vec, const {real_t}* __restrict__ z_vec,
            const {real_t}* __restrict__ kx, const {real_t}* __restrict__ ky, const {real_t}* __restrict__ kz, 
            const {real_t}* __restrict__ w, const {comp_t}* __restrict__ super_vec,
            {comp_t}* __restrict__ out,
            {real_t} z_scalar, {real_t} t, int nx, int num_pts, int num_waves
        ) {{
            int p = blockDim.x * blockIdx.x + threadIdx.x;
            if (p >= num_pts) return;

            {coord_logic.format(real_t=real_t)}
            
            {comp_t} acc[{num_components}];
            #pragma unroll
            for(int i=0; i<{num_components}; i++) acc[i] = {comp_t}(0, 0);

            for (int i = 0; i < num_waves; i++) {{
                {real_t} phase = kx[i]*px + ky[i]*py + kz[i]*pz - w[i]*t;
                {real_t} s, c;
                {sincos_f}(phase, &s, &c);
                {comp_t} wf(c, s);

                #pragma unroll
                for (int j = 0; j < {num_components}; j++) {{
                    acc[j] += super_vec[i * {num_components} + j] * wf;
                }}
            }}

            for (int j = 0; j < {num_components}; j++) {{
                out[j * num_pts + p] = acc[j];
            }}
        }}
        """
        kernel = cp.RawKernel(kernel_code, 'compute_kernel')
        self._kernel_cache[key] = kernel
        return kernel

    def _prepare_super_vec(self, need_b, need_derivs):
        vecs = [self.c_base]
        if need_b:
            bx = (self.ky * self.c_base[2] - self.kz * self.c_base[1]) * self.inv_w
            by = (self.kz * self.c_base[0] - self.kx * self.c_base[2]) * self.inv_w
            bz = (self.kx * self.c_base[1] - self.ky * self.c_base[0]) * self.inv_w
            vecs.append(cp.stack([bx, by, bz]))
        if need_derivs:
            ik = 1j * cp.stack([self.kx, self.ky, self.kz])
            for i in range(3): vecs.append(self.c_base * ik[i])
        
        return cp.ascontiguousarray(cp.vstack(vecs).T)

    def compute_grid(self, x_vec, y_vec, z, t, need_b=True, need_derivs=True, progress_callback=None):
        nx, ny = len(x_vec), len(y_vec)
        
        super_vec = self._prepare_super_vec(need_b, need_derivs)
        num_comps = super_vec.shape[1]
        kernel = self._get_kernel(num_comps, is_grid=True)

        # Pre-allocate the final CPU array to store results
        out_h = np.zeros((num_comps, ny, nx), dtype=np.complex128)

        # Move the X vector to GPU once
        x_g = cp.asarray(x_vec, dtype=self.real_dt)

        # Target ~500MB maximum VRAM footprint for the output buffer
        bytes_per_element = 8 if self.use_single else 16
        bytes_per_row = nx * num_comps * bytes_per_element
        MAX_VRAM_BYTES = 1000 * 1024 * 1024 # 1 GB
        
        # Calculate how many rows we can safely process at once
        rows_per_batch = max(1, MAX_VRAM_BYTES // bytes_per_row)

        for i in range(0, ny, rows_per_batch):
            end = min(i + rows_per_batch, ny)
            cur_ny = end - i
            cur_pts = nx * cur_ny

            # Move just this chunk of Y coordinates to GPU
            y_g = cp.asarray(y_vec[i:end], dtype=self.real_dt)
            
            # Allocate GPU buffer for just this batch
            out_g = cp.empty((num_comps, cur_pts), dtype=self.comp_dt)

            threads = 256
            blocks = (cur_pts + threads - 1) // threads
            
            kernel((blocks,), (threads,), (
                x_g, y_g, None, self.kx, self.ky, self.kz, self.w, 
                super_vec, out_g, 
                self.real_dt(z), self.real_dt(t), cp.int32(nx), cp.int32(cur_pts), cp.int32(self.num_waves)
            ))

            # Fetch result to CPU and reshape directly into the pre-allocated CPU array
            out_h[:, i:end, :] = out_g.get().reshape(num_comps, cur_ny, nx)
            
            if progress_callback: 
                progress_callback(cur_ny)
    
        # Unpack the CPU array into standard shapes
        E = out_h[0:3]
        idx = 3
        B = out_h[idx:idx+3] if need_b else None
        if need_b: idx += 3
        D = (out_h[idx:idx+3], out_h[idx+3:idx+6], out_h[idx+6:idx+9]) if need_derivs else (None,None,None)
        
        return E, D, B
        
    def compute_cloud(self, x, y, z, t, need_b=True, need_derivs=True, progress_callback=None):
        num_pts = len(x)
        super_vec = self._prepare_super_vec(need_b, need_derivs)
        num_comps = super_vec.shape[1]
        kernel = self._get_kernel(num_comps, is_grid=False)

        E_h, B_h, dx_h, dy_h, dz_h = self._allocate_cpu_arrays((num_pts,))
        CHUNK = 500_000

        for s in range(0, num_pts, CHUNK):
            e = min(s + CHUNK, num_pts)
            cur_n = e - s
            x_g, y_g, z_g = [cp.asarray(arr[s:e], dtype=self.real_dt) for arr in [x,y,z]]
            out_g = cp.empty((num_comps, cur_n), dtype=self.comp_dt)

            kernel(((cur_n+255)//256,), (256,), (
                x_g, y_g, z_g, self.kx, self.ky, self.kz, self.w, 
                super_vec, out_g, 0.0, self.real_dt(t), 0, cp.int32(cur_n), cp.int32(self.num_waves)
            ))

            out_c = out_g.get()
            if progress_callback: progress_callback(cur_n)
            E_h[:, s:e] = out_c[0:3]
            idx = 3
            if need_b: B_h[:, s:e] = out_c[idx:idx+3]; idx += 3
            if need_derivs:
                dx_h[:, s:e], dy_h[:, s:e], dz_h[:, s:e] = out_c[idx:idx+3], out_c[idx+3:idx+6], out_c[idx+6:idx+9]
        
        return E_h, (dx_h, dy_h, dz_h) if need_derivs else (None,None,None), B_h if need_b else None

    def _allocate_cpu_arrays(self, shape):
        return tuple(np.zeros((3, *shape), dtype=np.complex128) for _ in range(5))

    def compute_point(self, x, y, z, t, need_b=True, need_derivs=True):
        E, D, B = self.compute_cloud(np.array([x]), np.array([y]), np.array([z]), t, need_b, need_derivs)
        return E[:,0], (tuple(d[:,0] for d in D) if need_derivs else (None,None,None)), (B[:,0] if need_b else None) # type: ignore
    
    def __del__(self):
        self._kernel_cache.clear()
