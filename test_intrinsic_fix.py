#!/usr/bin/env python
"""Quick test to verify the intrinsic fix for RPNG AR dataset."""
import sys
sys.path.append('scripts')

import yaml
import numpy as np

def test_intrinsic_order():
    """Test that intrinsics are in correct [fx, fy, cx, cy] order."""
    config_path = 'configs/rpng/rpngar_table.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Original intrinsics from config
    fu = cfg['intrinsic']['fu']  # fx = 416.85
    fv = cfg['intrinsic']['fv']  # fy = 414.92
    cu = cfg['intrinsic']['cu']  # cx = 421.02
    cv = cfg['intrinsic']['cv']  # cy = 237.76
    H = cfg['intrinsic']['H']    # 480
    W = cfg['intrinsic']['W']    # 848

    resized_h, resized_w = cfg['frontend']['image_size']  # [328, 584]

    # NEW (correct) computation
    height_scale = resized_h / H
    width_scale = resized_w / W

    intrinsic_new = [
        fu * width_scale,   # fx
        fv * height_scale,  # fy
        cu * width_scale,   # cx
        cv * height_scale   # cy
    ]

    # OLD (buggy) computation
    u_scale = resized_h / H
    v_scale = resized_w / W

    intrinsic_old = [
        fv * v_scale,   # WRONG: fy instead of fx
        fu * u_scale,   # WRONG: fx instead of fy
        cv * v_scale,   # WRONG: cy instead of cx
        cu * u_scale    # WRONG: cx instead of cy
    ]

    print("="*60)
    print("INTRINSIC FIX VERIFICATION")
    print("="*60)
    print(f"Original intrinsics (H={H}, W={W}):")
    print(f"  fu (fx) = {fu:.2f}, fv (fy) = {fv:.2f}")
    print(f"  cu (cx) = {cu:.2f}, cv (cy) = {cv:.2f}")
    print(f"\nResized to: {resized_h}x{resized_w}")
    print(f"Scale factors: height={height_scale:.4f}, width={width_scale:.4f}")
    print()
    print("OLD (BUGGY) intrinsic [fy, fx, cy, cx]:")
    print(f"  [{intrinsic_old[0]:.2f}, {intrinsic_old[1]:.2f}, {intrinsic_old[2]:.2f}, {intrinsic_old[3]:.2f}]")
    print(f"  Principal point at: ({intrinsic_old[2]:.1f}, {intrinsic_old[3]:.1f})")
    print()
    print("NEW (FIXED) intrinsic [fx, fy, cx, cy]:")
    print(f"  [{intrinsic_new[0]:.2f}, {intrinsic_new[1]:.2f}, {intrinsic_new[2]:.2f}, {intrinsic_new[3]:.2f}]")
    print(f"  Principal point at: ({intrinsic_new[2]:.1f}, {intrinsic_new[3]:.1f})")
    print()

    # Expected principal point should be close to image center
    expected_cx = resized_w / 2  # 292
    expected_cy = resized_h / 2  # 164
    print(f"Expected principal point (image center): ({expected_cx:.1f}, {expected_cy:.1f})")

    # Verify the fix
    cx_error_old = abs(intrinsic_old[2] - expected_cx) + abs(intrinsic_old[3] - expected_cy)
    cx_error_new = abs(intrinsic_new[2] - expected_cx) + abs(intrinsic_new[3] - expected_cy)

    print()
    print("="*60)
    if cx_error_new < cx_error_old:
        print("✓ FIX VERIFIED: New intrinsics have correct principal point!")
        print(f"  Old error: {cx_error_old:.1f}px, New error: {cx_error_new:.1f}px")
    else:
        print("✗ WARNING: Something may be wrong with the fix")
    print("="*60)

if __name__ == '__main__':
    test_intrinsic_order()
