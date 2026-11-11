# 订单与支付双表对账环境

仓库 data 目录下有两份数据：orders.csv 是订单状态记录（order_id,status,amount,updated_at），时间戳为运营系统落库的北京时间（无时区标记）；payments.csv 是支付流水（pay_id,order_id,amount,paid_at），时间戳带 UTC 标记。任务要求见题面。

运行约定：脚本从仓库根目录运行，数据用相对路径 data/。
