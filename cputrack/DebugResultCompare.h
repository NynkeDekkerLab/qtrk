#pragma once

#ifdef _DEBUG

extern std::vector< std::complex<float> > cmp_cpu_qi_fft_in;
extern std::vector< std::complex<float> > cmp_gpu_qi_fft_in;

extern std::vector< std::complex<float> > cmp_cpu_qi_fft_out;
extern std::vector< std::complex<float> > cmp_gpu_qi_fft_out;

extern std::vector< float > cmp_cpu_qi_prof;
extern std::vector< float > cmp_gpu_qi_prof;

#endif

