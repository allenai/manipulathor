from typing import Dict, List, Optional
from typing_extensions import Literal
import math

from manipulathor_utils.debugger_util import ForkedPdb
from omegaconf import DictConfig, OmegaConf
# from ai2thor.controller import Controller
from utils.stretch_utils.stretch_ithor_arm_environment import StretchManipulaTHOREnvironment
from ai2thor.util import metrics
from allenact.utils.cache_utils import DynamicDistanceCache
from allenact.utils.system import get_logger
from utils.procthor_utils.procthor_types import Vector3


# def get_reachable_positions_procthor(controller):
#     event = controller.step('GetReachablePositions')
#     reachable_positions = event.metadata['actionReturn']
#     return reachable_positions

PROCTHOR_INVALID_SCENES = [6392, 1570, 175, 3572, 6854, 3526, 4104, 3014, 4670, 1308, 228, 954, 6476, 4481, 5506, 3864, 3818, 6843, 1745, 2330, 6648, 6869, 4032, 5635, 2027, 4586, 994, 5618, 610, 3166, 1627, 4251, 2081, 6807, 6501, 964, 714, 712, 5408, 898, 4692, 362, 1881, 3446, 2299, 1042, 3139, 5942, 3326, 6222, 5788, 1380, 2577, 347, 6735, 4987, 775, 6026, 988, 6453, 5082, 1741, 2796, 2247, 1682, 5272, 297, 5155, 4937, 5662, 1326, 1729, 4835, 187, 6012, 5501, 1788, 4441, 4729, 6030, 5023, 1108, 6054, 3801, 3134, 740, 495, 2072, 1617, 6035, 2508, 6369, 2895, 4442, 726, 5482, 3214, 1724, 6313, 2382, 1950, 1644, 5649, 4498, 3355, 208, 2539, 1392, 2487, 6450, 3521, 810, 6886, 1590, 3739, 2670, 1294, 4798, 4376, 1385, 1421, 676, 5555, 3342, 5552, 6060, 3133, 1663, 756, 50, 5288, 1491, 2961, 141, 3456, 5193, 2834, 1847, 4714, 6493, 3748, 5863, 84, 5025, 5241, 3353, 5640, 1353, 336, 3055, 2451, 5217, 4767, 3670, 6730] \
+ \
[21, 41, 43, 49, 64, 106, 124, 155, 157, 160, 179, 210, 212, 215, 229, 232, 233, 245, 259, 307, 319, 338, 352, 353, 355, 365, 380, 392, 409, 430, 449, 450, 469, 471, 486, 489, 504, 531, 592, 616, 639, 641, 646, 652, 673, 674, 692, 715, 716, 723, 745, 789, 794, 809, 819, 829, 868, 887, 899, 911, 948, 975, 990, 996, 1003, 1018, 1021, 1024, 1035, 1075, 1097, 1106, 1114, 1122, 1137, 1178, 1209, 1237, 1246, 1254, 1262, 1270, 1319, 1332, 1335, 1362, 1368, 1388, 1393, 1401, 1418, 1443, 1469, 1502, 1514, 1519, 1522, 1549, 1613, 1662, 1681, 1697, 1703, 1738, 1742, 1796, 1906, 1913, 1924, 1929, 1941, 1965, 1978, 1984, 1999, 2010, 2056, 2057, 2077, 2079, 2083, 2100, 2104, 2118, 2150, 2166, 2175, 2181, 2192, 2270, 2275, 2327, 2335, 2359, 2395, 2410, 2417, 2420, 2433, 2441, 2457, 2459, 2470, 2546, 2552, 2589, 2638, 2639, 2644, 2649, 2650, 2664, 2684, 2696, 2731, 2738, 2764, 2765, 2849, 2851, 2946, 2955, 3004, 3012, 3023, 3047, 3067, 3070, 3077, 3079, 3113, 3129, 3131, 3160, 3161, 3170, 3172, 3179, 3193, 3198, 3200, 3205, 3242, 3260, 3285, 3300, 3301, 3363, 3382, 3425, 3434, 3439, 3447, 3466, 3471, 3487, 3488, 3522, 3537, 3545, 3563, 3564, 3663, 3679, 3718, 3731, 3749, 3762, 3771, 3775, 3793, 3869, 3873, 3881, 3904, 3905, 3922, 3925, 3940, 3954, 3964, 3970, 3988, 4002, 4011, 4015, 4021, 4037, 4079, 4123, 4147, 4162, 4192, 4201, 4223, 4244, 4259, 4286, 4333, 4357, 4366, 4416, 4468, 4475, 4478, 4489, 4492, 4494, 4509, 4524, 4537, 4539, 4544, 4549, 4550, 4558, 4576, 4606, 4609, 4656, 4674, 4676, 4701, 4711, 4731, 4734, 4782, 4790, 4815, 4816, 4823, 4866, 4900, 4907, 4912, 4935, 4962, 4966, 4972, 5008, 5012, 5030, 5033, 5051, 5064, 5074, 5096, 5113, 5121, 5154, 5173, 5174, 5216, 5220, 5224, 5237, 5242, 5249, 5253, 5267, 5282, 5305, 5322, 5331, 5341, 5352, 5357, 5368, 5370, 5380, 5392, 5399, 5405, 5470, 5472, 5480, 5498, 5521, 5574, 5596, 5601, 5605, 5607, 5634, 5638, 5645, 5669, 5676, 5683, 5728, 5732, 5742, 5743, 5748, 5750, 5762, 5770, 5779, 5821, 5833, 5849, 5856, 5866, 5874, 5910, 5918, 5930, 5948, 5957, 5967, 5969, 5981, 6007, 6017, 6063, 6079, 6086, 6103, 6108, 6135, 6146, 6159, 6176, 6203, 6210, 6212, 6221, 6236, 6260, 6280, 6286, 6290, 6308, 6315, 6341, 6371, 6419, 6456, 6457, 6550, 6553, 6573, 6603, 6619, 6621, 6631, 6643, 6653, 6663, 6670, 6690, 6722, 6727, 6742, 6767, 6802, 6815, 6857, 6903, 6929, 6945, 6958, 6970, 6971, 6981, 6982, 6983, 6988, 6996] + \
[5517,2172] + [5262, 4012, 4005, 5608, 3948, 2856, 2810, 4232, 5419, 1223, 1621]


def position_dist(
    p0: Vector3,
    p1: Vector3,
    ignore_y: bool = False,
    dist_fn: Literal["l1", "l2"] = "l2",
) -> float:
    """Distance between two points of the form {"x": x, "y": y, "z": z}."""
    if dist_fn == "l1":
        return (
            abs(p0["x"] - p1["x"])
            + (0 if ignore_y else abs(p0["y"] - p1["y"]))
            + abs(p0["z"] - p1["z"])
        )
    elif dist_fn == "l2":
        return math.sqrt(
            (p0["x"] - p1["x"]) ** 2
            + (0 if ignore_y else (p0["y"] - p1["y"]) ** 2)
            + (p0["z"] - p1["z"]) ** 2
        )
    else:
        raise NotImplementedError(
            'dist_fn must be in {"l1", "l2"}.' f" You gave {dist_fn}"
        )


def distance_to_object_id(
    env: StretchManipulaTHOREnvironment,
    distance_cache: DynamicDistanceCache,
    object_id: str,
    house_name: str,
) -> Optional[float]:
    """Minimal geodesic distance to object of given objectId from agent's
    current location.
    It might return -1.0 for unreachable targets.
    # TODO: return None for unreachable targets.
    """

    def path_from_point_to_object_id(
        point: Dict[str, float], object_id: str, allowed_error: float
    ) -> Optional[List[Dict[str, float]]]:
        event = controller.step(
            action="GetShortestPath",
            objectId=object_id,
            position=point,
            allowedError=allowed_error,
        )
        if event:
            return event.metadata["actionReturn"]["corners"]
        else:
            get_logger().debug(
                f"Failed to find path for {object_id} in {house_name}."
                f' Start point {point}, agent state {event.metadata["agent"]}.'
            )
            return None

    def distance_from_point_to_object_id(
        point: Dict[str, float], object_id: str, allowed_error: float
    ) -> float:
        """Minimal geodesic distance from a point to an object of the given
        type.
        It might return -1.0 for unreachable targets.
        """
        path = path_from_point_to_object_id(point, object_id, allowed_error)
        if path:
            # Because `allowed_error != 0` means that the path returned above might not start
            # at `point`, we explicitly add any offset there is.
            dist = position_dist(p0=point, p1=path[0], ignore_y=True)
            return metrics.path_distance(path) + dist
        return -1.0

    def retry_dist(position: Dict[str, float], object_id: str) -> float:
        allowed_error = 0.05
        debug_log = ""
        d = -1.0
        while allowed_error < 2.5:
            d = distance_from_point_to_object_id(position, object_id, allowed_error)
            if d < 0:
                debug_log = (
                    f"In house {house_name}, could not find a path from {position} to {object_id} with"
                    f" {allowed_error} error tolerance. Increasing this tolerance to"
                    f" {2 * allowed_error} any trying again."
                )
                allowed_error *= 2
            else:
                break
        if d < 0:
            get_logger().warning(
                f"In house {house_name}, could not find a path from {position} to {object_id}"
                f" with {allowed_error} error tolerance. Returning a distance of -1."
            )
        elif debug_log != "":
            get_logger().debug(debug_log)
        return d

    return distance_cache.find_distance(
        scene_name=house_name,
        position=controller.last_event.metadata["agent"]["position"],
        target=object_id,
        native_distance_function=retry_dist,
    )

def spl_metric(
    success: bool, optimal_distance: float, travelled_distance: float
) -> Optional[float]:
    # TODO: eventually should be -> float
    if optimal_distance < 0:
        # TODO: update when optimal_distance must be >= 0.
        # raise ValueError(
        #     f"optimal_distance must be >= 0. You gave: {optimal_distance}."
        # )
        # return None
        return 0.0
    elif not success:
        return 0.0
    elif optimal_distance == 0:
        return 1.0 if travelled_distance == 0 else 0.0
    else:
        return optimal_distance / max(travelled_distance, optimal_distance)
