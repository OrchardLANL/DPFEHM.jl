using Test
import DPFEHM

#test the Corey relative permeability functions against some known good values
N_n, N_w = 2.0, 2.0
S_nir, S_wir = 0.1, 0.2
k0_n, k0_w = 0.2142, 0.85
@test 0.0 ≈ DPFEHM.RelPerm.Corey.kr_n(0.0, S_wir, S_nir, N_n, k0_n)
@test 0.009835714285714288 ≈ DPFEHM.RelPerm.Corey.kr_n(0.25, S_wir, S_nir, N_n, k0_n)
@test 0.06994285714285717 ≈ DPFEHM.RelPerm.Corey.kr_n(0.5, S_wir, S_nir, N_n, k0_n)
@test 0.18469285714285716 ≈ DPFEHM.RelPerm.Corey.kr_n(0.75, S_wir, S_nir, N_n, k0_n)
@test 0.2142 ≈ DPFEHM.RelPerm.Corey.kr_n(1.0, S_wir, S_nir, N_n, k0_n)
@test 0.0 ≈ DPFEHM.RelPerm.Corey.kr_w(0.0, S_wir, S_nir, N_w, k0_w)
@test 0.004336734693877549 ≈ DPFEHM.RelPerm.Corey.kr_w(0.25, S_wir, S_nir, N_w, k0_w)
@test 0.15612244897959177 ≈ DPFEHM.RelPerm.Corey.kr_w(0.5, S_wir, S_nir, N_w, k0_w)
@test 0.5247448979591837 ≈ DPFEHM.RelPerm.Corey.kr_w(0.75, S_wir, S_nir, N_w, k0_w)
@test 0.85 ≈ DPFEHM.RelPerm.Corey.kr_w(1.0, S_wir, S_nir, N_w, k0_w)
