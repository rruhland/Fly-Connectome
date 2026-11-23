"""Opt-in collision filtering between the tabletop and moving SO-101 link shapes."""

# Dedicated mask keeps these link shapes out of ball scoring and self-contact.
_TABLE_MASK = 1 << 12
MOVING = {"learner_shoulder", "learner_upper_arm", "learner_elbow", "learner_wrist",
          "learner_gripper", "learner_wrist_camera", "learner_wrist_camera_mount",
          "learner_moving_jaw_so101_v1"}


def enable_table_contacts(document):
    world = document.find("worldbody")
    surfaces = [g for g in world.findall("geom") if g.get("name") in {"table", "table_right"}]
    for geom in surfaces:
        # TT4's table_collision default supplies type 2.
        geom.set("contype", str(int(geom.get("contype", "2")) | _TABLE_MASK))
    links = []
    for body in world.iter("body"):
        if body.get("name") not in MOVING:
            continue
        for geom in body.findall("geom"):
            # The imported collision and visual elements use the same CAD meshes.
            # TT19 retains the visual instance. Give it a table-only affinity.
            if geom.get("class") == "visual" or geom.get("name") in {
                "tt20_camera_boom", "tt20_rgb_housing", "tt20_rgb_barrel"
            }:
                geom.set("conaffinity", str(int(geom.get("conaffinity", "0")) | _TABLE_MASK))
                links.append(geom)
    if not surfaces or not links:
        raise ValueError("Missing table or moving arm geometry")
    return document
