"""
完整流程验证：动态阈值计算和应用
"""

def verify_complete_flow():
    """验证从config到loss的完整数据流"""
    print("=" * 70)
    print("完整流程验证：动态阈值计算和应用")
    print("=" * 70)

    checks = []

    # 1. 检查 config.py
    print("\n[1/5] 检查 config.py - LossConfig定义")
    print("-" * 70)
    with open('myGNN/config.py', 'r', encoding='utf-8') as f:
        config_content = f.read()

    has_use_dynamic = 'self.use_dynamic_threshold' in config_content
    has_alert_temp = 'self.alert_temp' in config_content
    has_doc = '高温阈值计算方式' in config_content

    print(f"  use_dynamic_threshold参数: {'存在' if has_use_dynamic else '缺失'}")
    print(f"  alert_temp参数: {'存在' if has_alert_temp else '缺失'}")
    print(f"  文档说明: {'完整' if has_doc else '缺失'}")

    checks.append(has_use_dynamic and has_alert_temp and has_doc)

    # 2. 检查 dataset.py - 两个数据加载函数
    print("\n[2/5] 检查 dataset.py - 计算ta_p90")
    print("-" * 70)
    with open('myGNN/dataset.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 检查_create_dataloaders_original
    orig_has_calc = False
    orig_has_stats = False
    orig_order_ok = False

    for i, line in enumerate(lines):
        if 'def _create_dataloaders_original(' in line:
            section = ''.join(lines[i:i+100])
            orig_has_calc = 'ta_p90 = float(np.percentile(' in section
            orig_has_stats = "'ta_p90': ta_p90" in section

            # 检查顺序
            ta_p90_line = None
            norm_line = None
            for j in range(i, min(i+100, len(lines))):
                if 'ta_p90 = ' in lines[j] and ta_p90_line is None:
                    ta_p90_line = j
                if 'MetData[:, :, :26] = ' in lines[j] and 'feature_mean' in lines[j]:
                    norm_line = j
                    break
            orig_order_ok = ta_p90_line is not None and norm_line is not None and ta_p90_line < norm_line
            break

    print(f"  _create_dataloaders_original:")
    print(f"    计算ta_p90: {'是' if orig_has_calc else '否'}")
    print(f"    添加到stats: {'是' if orig_has_stats else '否'}")
    print(f"    计算顺序: {'正确(标准化前)' if orig_order_ok else '错误'}")

    # 检查_create_dataloaders_separated
    sep_has_calc = False
    sep_has_stats = False
    sep_order_ok = False

    for i, line in enumerate(lines):
        if 'def _create_dataloaders_separated(' in line:
            section = ''.join(lines[i:i+200])
            sep_has_calc = 'ta_p90 = float(np.percentile(' in section
            sep_has_stats = "'ta_p90': ta_p90" in section

            # 检查顺序
            ta_p90_line = None
            norm_line = None
            for j in range(i, min(i+200, len(lines))):
                if 'ta_p90 = ' in lines[j] and ta_p90_line is None:
                    ta_p90_line = j
                if 'MetData[:, :, static_indices] = ' in lines[j] and norm_line is None:
                    norm_line = j
                if ta_p90_line and norm_line:
                    break
            sep_order_ok = ta_p90_line is not None and norm_line is not None and ta_p90_line < norm_line
            break

    print(f"  _create_dataloaders_separated:")
    print(f"    计算ta_p90: {'是' if sep_has_calc else '否'}")
    print(f"    添加到stats: {'是' if sep_has_stats else '否'}")
    print(f"    计算顺序: {'正确(标准化前)' if sep_order_ok else '错误'}")

    checks.append(orig_has_calc and orig_has_stats and orig_order_ok)
    checks.append(sep_has_calc and sep_has_stats and sep_order_ok)

    # 3. 检查 train.py - 应用动态阈值
    print("\n[3/5] 检查 train.py - 应用动态阈值")
    print("-" * 70)
    with open('myGNN/train.py', 'r', encoding='utf-8') as f:
        train_content = f.read()

    has_get_ta_p90 = "ta_p90 = stats['ta_p90']" in train_content
    has_check = 'if config.loss_config.use_dynamic_threshold:' in train_content
    has_update = 'config.loss_config.alert_temp = ta_p90' in train_content

    print(f"  获取ta_p90: {'是' if has_get_ta_p90 else '否'}")
    print(f"  检查use_dynamic_threshold: {'是' if has_check else '否'}")
    print(f"  更新alert_temp: {'是' if has_update else '否'}")

    # 检查执行顺序
    get_pos = train_content.find("ta_p90 = stats['ta_p90']")
    update_pos = train_content.find("config.loss_config.alert_temp = ta_p90")
    loss_pos = train_content.find("criterion = get_loss_function(config)")

    order_ok = get_pos < update_pos < loss_pos
    print(f"  执行顺序: {'正确(获取→更新→创建loss)' if order_ok else '错误'}")

    checks.append(has_get_ta_p90 and has_check and has_update and order_ok)

    # 4. 检查 train_enhanced.py - 损失函数创建
    print("\n[4/5] 检查 train_enhanced.py - 损失函数工厂")
    print("-" * 70)
    with open('myGNN/train_enhanced.py', 'r', encoding='utf-8') as f:
        enhanced_content = f.read()

    has_factory = 'def get_loss_function(config):' in enhanced_content
    uses_alert_temp = 'alert_temp=loss_cfg.alert_temp' in enhanced_content
    passes_ta_stats = 'ta_mean=config.ta_mean' in enhanced_content and 'ta_std=config.ta_std' in enhanced_content

    print(f"  工厂函数存在: {'是' if has_factory else '否'}")
    print(f"  使用alert_temp参数: {'是' if uses_alert_temp else '否'}")
    print(f"  传递ta_mean/ta_std: {'是' if passes_ta_stats else '否'}")

    checks.append(has_factory and uses_alert_temp and passes_ta_stats)

    # 5. 检查 losses.py - WeightedTrendMSELoss
    print("\n[5/5] 检查 losses.py - WeightedTrendMSELoss")
    print("-" * 70)
    with open('myGNN/losses.py', 'r', encoding='utf-8') as f:
        losses_content = f.read()

    has_class = 'class WeightedTrendMSELoss(nn.Module):' in losses_content
    has_init_param = 'alert_temp' in losses_content and '__init__' in losses_content
    uses_in_forward = 'self.alert_temp' in losses_content and 'forward' in losses_content

    print(f"  类定义存在: {'是' if has_class else '否'}")
    print(f"  __init__接收alert_temp: {'是' if has_init_param else '否'}")
    print(f"  forward中使用alert_temp: {'是' if uses_in_forward else '否'}")

    checks.append(has_class and has_init_param and uses_in_forward)

    # 总结
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)

    steps = [
        "config.py - LossConfig定义",
        "dataset.py - _create_dataloaders_original",
        "dataset.py - _create_dataloaders_separated",
        "train.py - 应用动态阈值",
        "train_enhanced.py - 损失函数工厂",
        "losses.py - WeightedTrendMSELoss"
    ]

    all_passed = all(checks)

    for i, (step, passed) in enumerate(zip(steps, checks)):
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {i+1}. {status} {step}")

    print("\n" + "=" * 70)
    if all_passed:
        print("[SUCCESS] 所有检查通过！动态阈值流程完整正确")
        print("=" * 70)
        print("\n流程说明:")
        print("  1. config.py: 定义use_dynamic_threshold和alert_temp")
        print("  2. dataset.py: 在标准化前计算训练集的ta_p90")
        print("  3. train.py: 根据use_dynamic_threshold更新alert_temp")
        print("  4. train_enhanced.py: 使用更新后的alert_temp创建损失函数")
        print("  5. losses.py: WeightedTrendMSELoss使用alert_temp计算权重")
        return True
    else:
        print("[FAIL] 存在问题，需要修复")
        print("=" * 70)
        return False


if __name__ == "__main__":
    try:
        success = verify_complete_flow()
        if not success:
            exit(1)
    except Exception as e:
        print(f"\n[ERROR] 发生错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
