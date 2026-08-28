"""Convert spider.urdf -> USD. Run: python spider/convert_spider.py --headless"""
import argparse, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a = p.parse_args()
app = AppLauncher(a).app
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
here = os.path.dirname(os.path.abspath(__file__))
cfg = UrdfConverterCfg(
    asset_path=os.path.join(here, "assets", "spider.urdf"),
    usd_dir=os.path.join(here, "assets", "spider_usd"),
    fix_base=False, merge_fixed_joints=False, force_usd_conversion=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=25.0, damping=1.0), target_type="position"),
)
print("[convert] USD:", UrdfConverter(cfg).usd_path)
app.close()
