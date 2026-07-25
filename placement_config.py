"""Placement config for the next dataset (baked 2026-07-24).

Defines the full site catalog (union of the confusers' reviewed 26-site set + the threats'
45-site set), registers it into anatomy.SITES, loads the objects_v5 registry, and maps every
object to its allowed sites:
  * confusers -> their reviewed OK sites (confuser_site_map.json)   [26-site space]
  * threats   -> ALL 45 sites                                        [45-site space]

Azimuth convention (anatomy): 0=front, +90=model's LEFT(+x), 180=back, -90=model's RIGHT(-x).
"""
import os, json
import anatomy
from anatomy import BELT, ANKLE

_HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.join(_HERE, "objects_v5")
CONFUSER_MAP = next((p for p in (os.path.join(_HERE, "confuser_site_map.json"),
                     "/Users/jonah/.claude/jobs/dab08409/tmp/confuser_site_map.json")
                     if os.path.exists(p)), os.path.join(_HERE, "confuser_site_map.json"))

# ---- full site catalog: name -> (region, facing|azimuth, z_lo, z_hi) ----
SITE_SPECS = {
    # waistband ring (beltline, BELT height band)
    "wb_front_mid": ("beltline", 0.0, *BELT), "wb_front_l": ("beltline", 45.0, *BELT),
    "wb_front_r": ("beltline", -45.0, *BELT), "wb_l": ("beltline", 90.0, *BELT),
    "wb_r": ("beltline", -90.0, *BELT), "wb_back_l": ("beltline", 135.0, *BELT),
    "wb_back_r": ("beltline", -135.0, *BELT), "wb_back_mid": ("beltline", 180.0, *BELT),
    # flank (side of midriff)
    "flank_l": ("abdomen", "side_l", None, None), "flank_r": ("abdomen", "side_r", None, None),
    # midline
    "belly": ("abdomen", "front", None, None), "chest": ("chest", "front", None, None),
    "small_of_back": ("lower_back", "back", None, None), "upper_back": ("upper_back", "back", None, None),
    # thigh: front/back/inner/outer (inner = toward body centre)
    "thigh_front_l": ("left_thigh", "front", None, None), "thigh_front_r": ("right_thigh", "front", None, None),
    "thigh_back_l": ("left_thigh", "back", None, None), "thigh_back_r": ("right_thigh", "back", None, None),
    "thigh_inner_l": ("left_thigh", -90.0, None, None), "thigh_inner_r": ("right_thigh", 90.0, None, None),
    "thigh_outer_l": ("left_thigh", 90.0, None, None), "thigh_outer_r": ("right_thigh", -90.0, None, None),
    # ankle (ANKLE band); inside = toward centre, outside = away
    "ankle_inside_l": ("left_shin", -90.0, *ANKLE), "ankle_inside_r": ("right_shin", 90.0, *ANKLE),
    "ankle_outside_l": ("left_shin", 90.0, *ANKLE), "ankle_outside_r": ("right_shin", -90.0, *ANKLE),
    "ankle_l": ("left_shin", -90.0, *ANKLE), "ankle_r": ("right_shin", 90.0, *ANKLE),  # confuser aliases = inside
    # shin: front (confuser) + back/inner/outer (threat), whole-shin region
    "shin_l": ("left_shin", "front", None, None), "shin_r": ("right_shin", "front", None, None),
    "shin_back_l": ("left_shin", "back", None, None), "shin_back_r": ("right_shin", "back", None, None),
    "shin_inner_l": ("left_shin", -90.0, None, None), "shin_inner_r": ("right_shin", 90.0, None, None),
    "shin_outer_l": ("left_shin", 90.0, None, None), "shin_outer_r": ("right_shin", -90.0, None, None),
    # arms
    "upper_arm_l": ("left_upper_arm", "any", None, None), "upper_arm_r": ("right_upper_arm", "any", None, None),
    "forearm_l": ("left_forearm", "any", None, None), "forearm_r": ("right_forearm", "any", None, None),
}
anatomy.SITES.update(SITE_SPECS)   # register so site_faces / sample_placement resolve them

# ---- the 45 (36-leaf) threat sites: threats go everywhere ----
THREAT_SITES = [
    "wb_front_mid","wb_front_l","wb_front_r","flank_l","flank_r","wb_back_mid","wb_back_l","wb_back_r",
    "wb_l","wb_r","belly","chest","small_of_back","upper_back",
    "thigh_front_l","thigh_front_r","thigh_back_l","thigh_back_r","thigh_inner_l","thigh_inner_r",
    "thigh_outer_l","thigh_outer_r","ankle_inside_l","ankle_inside_r","ankle_outside_l","ankle_outside_r",
    "upper_arm_l","upper_arm_r","forearm_l","forearm_r","shin_back_l","shin_back_r",
    "shin_inner_l","shin_inner_r","shin_outer_l","shin_outer_r",
]

# ---- object registry (objects_v5 manifest) ----
OBJECTS = {o["name"]: o for o in json.load(open(os.path.join(V5, "manifest.json")))}
OBJECTS["gun1_glock17"]["two_material"] = True   # steel slide + polymer frame via threats._glock_parts (dataset4)

# ---- placement map ----
def _build_placement():
    pm = {}
    cmap = json.load(open(CONFUSER_MAP))
    for name, o in OBJECTS.items():
        if o["kind"] == "threat":
            pm[name] = list(THREAT_SITES)                 # threats -> all 45
        else:                                             # confusers -> reviewed OK sites
            key = "phone" if name in ("phone_nokia", "phone_modern") else name
            pm[name] = list(cmap.get(key, {}).get("ok", []))
    return pm

PLACEMENT = _build_placement()

if __name__ == "__main__":
    print(f"sites in catalog: {len(SITE_SPECS)}  |  threat sites: {len(THREAT_SITES)}")
    print(f"objects: {sum(o['kind']=='threat' for o in OBJECTS.values())} threat / "
          f"{sum(o['kind']=='benign' for o in OBJECTS.values())} benign")
    for n, sites in PLACEMENT.items():
        print(f"  {n:20s} {OBJECTS[n]['kind']:6s} -> {len(sites):2d} sites")
