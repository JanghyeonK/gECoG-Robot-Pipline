import numpy as np

def map_behavior_labels(behavior_series):
    """Convert Behavior text to label index (0~6)."""
    behavior = behavior_series.str.lower().str.replace("[']", "", regex=True)
    behavior = behavior.str.replace("−", "-", regex=False)
    behavior = behavior.str.replace("-", " ", regex=False)
    behavior = behavior.str.replace("foward", "forward")
    behavior = behavior.str.replace("stting", "sitting")

    labels = np.full(len(behavior), np.nan)
    is_nonchew = behavior.str.contains("non chewing")
    is_chew = behavior.str.contains("chewing") & (~is_nonchew)
    is_walk = behavior.str.contains("walking")
    is_sit = behavior.str.contains("sitting")
    is_left = behavior.str.contains("turning left")
    is_right = behavior.str.contains("turning right")

    labels[(is_chew & is_walk)] = 0
    labels[(is_chew & is_sit)] = 1
    labels[(is_nonchew & is_walk & ~(is_left | is_right))] = 2
    labels[(is_nonchew & is_walk & is_left)] = 3
    labels[(is_nonchew & is_sit & ~(is_left | is_right))] = 4
    labels[(is_nonchew & is_sit & is_left)] = 5
    labels[(is_nonchew & is_sit & is_right)] = 6

    bad_idx = np.where(np.isnan(labels))[0]
    return labels.astype("float"), bad_idx


def map_labels_to_ecog_time(ecog_time, label_time, label_part):
    """Expand interval labels to ECoG sample-wise labels."""
    t_start, t_end, label_value = [], [], []
    for s, lbl in zip(label_time, label_part):
        s = str(s).replace("−", "-").replace("~", "-").replace(" ", "")
        try:
            a, b = s.split("-")
            t_start.append(float(a))
            t_end.append(float(b))
            label_value.append(lbl)
        except Exception:
            continue

    label_full = np.zeros_like(ecog_time) + 4.0
    for a, b, lbl in zip(t_start, t_end, label_value):
        i0 = np.searchsorted(ecog_time, a, side="left")
        i1 = np.searchsorted(ecog_time, b, side="right")
        label_full[i0:i1] = lbl
    return label_full
