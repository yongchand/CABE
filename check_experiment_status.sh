#!/bin/bash
# 检查L-BFGS重试实验的状态

echo "=============================================="
echo "L-BFGS 重试实验状态检查"
echo "=============================================="
echo ""

# 检查进程是否还在运行
if ps aux | grep -v grep | grep "run_ablation_experiments.py" > /dev/null; then
    echo "✅ 实验正在运行中..."
    ps aux | grep -v grep | grep "run_ablation_experiments.py" | head -1
    echo ""
else
    echo "❌ 实验已完成或未运行"
    echo ""
fi

# 检查已创建的实验目录
echo "已创建的实验目录："
echo "-----------------------------------"
ls -1 experiments_lbfgs_retry/ 2>/dev/null | grep -E "^MoNIG" | while read dir; do
    if [ -f "experiments_lbfgs_retry/$dir/training.log" ]; then
        # 检查是否成功（有模型文件）
        if [ -f "experiments_lbfgs_retry/$dir/best_"*".pt" ]; then
            echo "  ✅ $dir - 训练完成"
        else
            # 检查是否有错误
            if grep -q "error\|Error\|ERROR\|killed\|Killed\|KILLED\|abort\|Abort\|ABORT" "experiments_lbfgs_retry/$dir/training.log" 2>/dev/null; then
                echo "  ❌ $dir - 训练失败"
            else
                echo "  🔄 $dir - 训练中..."
            fi
        fi
    else
        echo "  ⏳ $dir - 等待开始"
    fi
done

echo ""
echo "实验进度统计："
echo "-----------------------------------"
total_dirs=$(ls -1 experiments_lbfgs_retry/ 2>/dev/null | grep -E "^MoNIG" | wc -l)
completed=$(find experiments_lbfgs_retry/ -name "best_*.pt" 2>/dev/null | wc -l)
echo "  总实验数: 9"
echo "  已创建目录: $total_dirs"
echo "  已完成: $completed"
echo "  剩余: $((9 - completed))"

# 检查结果CSV文件
echo ""
echo "结果文件："
echo "-----------------------------------"
if [ -f "experiments_lbfgs_retry/ablation_results_"*".csv" ]; then
    result_file=$(ls -t experiments_lbfgs_retry/ablation_results_*.csv 2>/dev/null | head -1)
    echo "  ✅ $result_file"
    successful=$(grep -c ",True," "$result_file" 2>/dev/null || echo 0)
    echo "  成功实验数: $successful"
else
    echo "  ⏳ 结果文件尚未生成"
fi

echo ""
echo "=============================================="
if [ $completed -eq 9 ]; then
    echo "🎉 所有实验已完成！可以运行分析了。"
    echo ""
    echo "运行以下命令进行分析："
    echo "  bash analyze_lbfgs_results.sh"
else
    echo "⏳ 实验进行中，请稍后再检查。"
    echo ""
    echo "再次运行此脚本检查状态："
    echo "  bash check_experiment_status.sh"
fi
echo "=============================================="

