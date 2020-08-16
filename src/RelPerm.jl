module RelPerm

module Corey
	function S_wn(S_w, S_wir, S_nir)
		return (S_w - S_wir) / (1 - S_wir - S_nir)
	end
	function kr_w(S_w, S_wir, S_nir, N_w, k0_w)
		return k0_w * (S_wn(S_w, S_wir, S_nir)) ^ N_w
	end
	function kr_n(S_n, S_wir, S_nir, N_n, k0_n)
		return k0_n * (1 - S_wn(1 - S_n, S_wir, S_nir)) ^ N_n
	end
end

end
