# Provenance

This repository was prepared from the author's supplied CIUM-Net archive and manuscript.
The source archive did not include a project-wide LICENSE file. This cleanup does not
assign a license to code or grant rights on behalf of upstream authors.

- HVI/PHVIT, encoder/decoder sampling, CAB and IEL descend from the HVI/CIDNet
  implementation retained in the supplied archive (manuscript reference 3:
  Q. Yan et al., *HVI: A New Color Space for Low-Light Image Enhancement*, CVPR 2025).
  The manuscript also credits Restormer for the channel-attention design (reference 30).
- The supplied `DIEM.py` contains the bidirectional gated mixing implementation now
  organized as `model.py` (BiCIU); the manuscript discusses ICLR/DIEM as related work
  (reference 4). Renaming it does not erase its source history.
- The Laplacian edge and Gaussian SSIM losses are adapted from the supplied loss code.
- `metrics_core.py` and `niqe_pris_params.npz` are retained from the archive's NIQE
  implementation, whose comments describe the BasicSR/MATLAB-style evaluation convention.
  Required statistical parameters are retained; unrelated conversion helpers were removed.
- LPIPS is an external dependency credited to R. Zhang et al., CVPR 2018 (reference 37).
  NIQE and BRISQUE are credited to A. Mittal and collaborators (references 38 and 39).
- BRISQUE uses the external `image-quality` 1.2.7 package, distributed under Apache 2.0,
  with a local API-compatibility adapter. Its model remains part of that dependency,
  not an enhancement checkpoint bundled with this repository.

Confirm and carry forward the applicable upstream licenses when assigning a public
project license. The compact project retains these notices because attribution is
part of the relevant implementation, not an unrelated experiment file.
