# DSER++ Learned BASR V2

基于 MrLi002/DSER-QQ 的完整修订源码。

本版修正粗对齐残差预测、直接检索监督及 RGB 来源判断证据，加入 DCN 调制弱先验与训练监控 CSV。

请先阅读 [MODIFICATION_GUIDE.md](MODIFICATION_GUIDE.md)，其中有逐文件修改说明、完整训练/迁移/续训/测试命令及验证边界。

- 原版与 V1 权重迁移使用 `--pretrained`，会重新初始化新版 BASR。
- V1 权重不能通过 `--resume` 直接变成 V2。
- 保留 `--basr-mode original` 对照。
- 完整文本源码、差异补丁和测试已包含；不含数据、模型权重及演示图片。

来源、许可证和版本见 BASE_COMMIT.txt、README_upstream.md、LICENSE、VERSION.txt。
本代码尚未完成真实数据完整训练。
