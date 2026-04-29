"""Launch prosperity3bt with HYDROGEL/VEV products injected into LIMITS."""
import sys
from prosperity3bt.data import LIMITS

# Round 3 "Gloves Off" 产品
LIMITS["HYDROGEL_PACK"] = 200
LIMITS["VELVETFRUIT_EXTRACT"] = 200
for K in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]:
    LIMITS[f"VEV_{K}"] = 300

from prosperity3bt.__main__ import main
main()
