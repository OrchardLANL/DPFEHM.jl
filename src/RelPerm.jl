module RelPerm

#TODO add capillary pressure here -- see equations 8.11 and 8.12 in "introduction to reservoir simulation using matlab/gnu octave" for the brooks-corey and van genuchten forms of capillary pressure

module Corey
	function S_wn(S_w, S_wir, S_nir)
		return (S_w - S_wir) / (1 - S_wir - S_nir)
	end
	function kr_w(S_w, S_wir, S_nir, N_w, k0_w)
		if S_w < S_wir
			return 0.0
		elseif S_w > 1 - S_nir
			return k0_w
		else
			return k0_w * (S_wn(S_w, S_wir, S_nir)) ^ N_w
		end
		#return k0_w * (S_wn(S_w, S_wir, S_nir)) ^ N_w
	end
	function kr_n(S_n, S_wir, S_nir, N_n, k0_n)
		if S_n < S_nir
			return 0.0
		elseif S_n > 1 - S_wir
			return k0_n
		else
			return k0_n * (1 - S_wn(1 - S_n, S_wir, S_nir)) ^ N_n
		end
		#return k0_n * (1 - S_wn(1 - S_n, S_wir, S_nir)) ^ N_n
	end
end

end
