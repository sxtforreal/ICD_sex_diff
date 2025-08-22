# Commit Messages for PR

## Main Commit
```
fix: 解决PyTorch Lightning重复日志记录错误

- 统一_step方法中的日志记录逻辑
- 移除validation_step中的重复self.log调用
- 根据stage参数调整日志记录行为
- 为splitdata模块使用不同的键名避免冲突

修复错误: You called `self.log(val/focal_loss, ...)` twice in `validation_step` with different arguments

Closes: #[issue_number]
```

## Detailed Commits (if needed)

### Commit 1: 重构_step方法
```
refactor: 重构_step方法统一处理日志记录

- 根据stage参数(train/val/test)调整日志记录参数
- 使用log_dict统一记录所有损失指标
- 训练阶段: on_step=True, on_epoch=True
- 验证阶段: on_step=False, on_epoch=True
```

### Commit 2: 简化validation_step
```
fix: 简化validation_step避免重复日志记录

- 移除validation_step中所有self.log调用
- 只调用_step方法并返回必要信息
- 避免val/focal_loss被记录两次的问题
```

### Commit 3: 处理模块冲突
```
feat: 为splitdata模块添加独立键名支持

- splitdata指标使用val/split_*前缀
- 避免与主要指标键名冲突
- 预防未来模块间的日志记录冲突
```

## Git Commands

```bash
# 1. 创建新分支
git checkout -b fix/duplicate-logging-error

# 2. 添加修改的文件
git add main/model.py

# 3. 提交修改
git commit -m "fix: 解决PyTorch Lightning重复日志记录错误

- 统一_step方法中的日志记录逻辑  
- 移除validation_step中的重复self.log调用
- 根据stage参数调整日志记录行为
- 为splitdata模块使用不同的键名避免冲突

修复错误: You called \`self.log(val/focal_loss, ...)\` twice in \`validation_step\` with different arguments"

# 4. 推送到远程
git push origin fix/duplicate-logging-error

# 5. 创建Pull Request
# 使用GitHub CLI (如果安装了)
gh pr create --title "Fix: 解决PyTorch Lightning重复日志记录错误" --body-file PULL_REQUEST_DESCRIPTION.md

# 或者在GitHub网页界面创建PR
```

## PR Title Options

1. `Fix: 解决PyTorch Lightning重复日志记录错误`
2. `fix(model): 统一日志记录避免重复调用错误`
3. `Fix duplicate logging error in validation_step`
4. `Refactor: 统一_step方法日志记录逻辑`

## Branch Name Options

- `fix/duplicate-logging-error`
- `fix/pytorch-lightning-logging`
- `refactor/unified-step-logging`
- `bugfix/val-focal-loss-duplicate`