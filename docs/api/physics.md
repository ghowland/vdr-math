<!-- markdownlint-disable -->

<a href="https://github.com/ghowland/vdr-math/blob/main/vdr/physics/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `physics`
vdr.physics — Exact physical computation domains. 

 from vdr.physics.qed import a2_coefficient  from vdr.physics.quantum import pauli_multiply, spin_rotation  from vdr.physics.orbital import kepler_newton  from vdr.physics.optics import system_matrix, verify_symplecticity 

VDR changes arithmetic, not physics. Does not derive new results. Shows routine physical computations produce exact results in VDR. 

Conservation laws verified by exact equality, not residual tolerance:  probability = 1, unitarity U†U = I, symplecticity det(M) = 1,  energy conservation, orbit closure. 





---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
