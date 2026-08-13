#!/bin/bash

# ==================== 配置区域 ====================
# 需要更新的分支列表（按需修改）
BRANCHES=("v2.7.1" "v2.10.0")

# 子模块路径（默认使用 op-plugin）
SUBMODULE_PATH="third_party/op-plugin"

# 时间戳格式：年月日时分
TIMESTAMP=$(date +"%Y%m%d%H%M")
# ================================================

echo "=========================================="
echo "开始更新 commit ID"
echo "时间戳: $TIMESTAMP"
echo "分支列表: ${BRANCHES[*]}"
echo "子模块: $SUBMODULE_PATH"
echo "=========================================="
echo ""

# 从 SUBMODULE_PATH 提取最后一个目录名作为模块名
MODULE_NAME=$(basename "$SUBMODULE_PATH")

for BRANCH in "${BRANCHES[@]}"; do
    echo "----------------------------------------"
    echo "处理分支: $BRANCH"
    echo "----------------------------------------"
    
    # 1. 检出原分支并拉取最新代码
    echo "→ 检出分支: $BRANCH"
    git checkout "$BRANCH" || { echo "❌ 分支 $BRANCH 不存在，跳过"; continue; }
    
    echo "→ 拉取最新代码"
    git pull origin "$BRANCH"
    
    # 2. 创建 update 分支
    UPDATE_BRANCH="${BRANCH}-update_${TIMESTAMP}"
    echo "→ 创建更新分支: $UPDATE_BRANCH"
    git checkout -b "$UPDATE_BRANCH"
    
    # 3. 初始化并更新子模块
    echo "→ 更新子模块: $SUBMODULE_PATH"
    git submodule init
    git submodule update --remote "$SUBMODULE_PATH"
    
    # 4. 仅添加子模块变更
    echo "→ 添加变更内容"
    git add "$SUBMODULE_PATH"
    
    # 5. 检查是否有变更
    if git diff --cached --quiet; then
        echo "⚠️  无变更，跳过提交"
        git checkout "$BRANCH"
        git branch -D "$UPDATE_BRANCH"
        continue
    fi
    
    # 6. 提交变更
    echo "→ 提交变更"
    git commit -m "Update ${MODULE_NAME} commit id"
    
    # 7. 推送到远程并捕获输出
    echo "→ 推送分支到远程"
    PUSH_OUTPUT=$(git push --set-upstream origin "$UPDATE_BRANCH" 2>&1)
    echo "$PUSH_OUTPUT"
    
    # 8. 从 push 输出中提取 URL
    REMOTE_URL=""
    
    # 尝试从 git push 输出中提取远程仓库 URL
    if [[ $PUSH_OUTPUT =~ (https?://[^[:space:]]+) ]]; then
        REMOTE_URL="${BASH_REMATCH[1]}"
        # 清理可能的末尾标点（但保留 ? 和 &）
        REMOTE_URL=$(echo "$REMOTE_URL" | sed 's/[[:punct:]]*$//' | sed 's/\.$//')
        
        # 如果是 GitCode 且缺少 target_branch 参数，则添加
        if [[ $REMOTE_URL =~ gitcode\.com.*merge_requests/new ]] && [[ ! $REMOTE_URL =~ target_branch= ]]; then
            REMOTE_URL="${REMOTE_URL}&target_branch=${BRANCH}"
        fi
    fi
    
    if [ -n "$REMOTE_URL" ]; then
        echo "→ 打开 MR/PR 创建页面: $REMOTE_URL"
        start "$REMOTE_URL" 2>/dev/null || open "$REMOTE_URL" 2>/dev/null || xdg-open "$REMOTE_URL" 2>/dev/null || echo "请手动打开: $REMOTE_URL"
    else
        echo "⚠️  无法获取 MR/PR 链接，请手动创建"
    fi
    
    echo "✅ 分支 $BRANCH 已完成"
    echo ""
done

echo "=========================================="
echo "所有分支推送完成，记得处理PR！"
echo "=========================================="
