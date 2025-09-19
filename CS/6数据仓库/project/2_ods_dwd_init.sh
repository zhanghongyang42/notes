#!/bin/bash

#业务库和神策库首次导入的时间
dt_casmart_first='20220218'
dt_sensorslog_first='20220218'

# ----------------------------------------------------------------------------------------------------------------------


product_favorites_create1="
use dwd;
DROP TABLE IF EXISTS dwd_user_member_product_favorites_full;
CREATE TABLE dwd_user_member_product_favorites_full(
                                                  "'`favorites_id`'" STRING COMMENT '收藏夹id',
                                                  "'`member_id`'" STRING COMMENT '会员id',
                                                  "'`product_id`'" STRING COMMENT '商品ID',
                                                  "'`add_time`'" STRING COMMENT '添加时间',
                                                  "'`create_time`'" STRING COMMENT '创建时间',
                                                  "'`operate_time`'" STRING COMMENT '操作时间'
) COMMENT '商品收藏事实表，周期性事实表，每日全量加载'
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

product_favorites1="
insert overwrite table dwd.dwd_user_member_product_favorites_full partition(dt=$dt_casmart_first)
select
    id,
    member_id,
    target_id,
    add_time,
    created,
    modified
from ods.ods_sys_tbl_member_favorites_upsert
where dt=$dt_casmart_first;
"

supplier_favorites_create2="
use dwd;
DROP TABLE IF EXISTS dwd_user_member_supplier_favorites_full;
CREATE TABLE dwd_user_member_supplier_favorites_full(
                                                   "'`favorites_id`'" STRING COMMENT '收藏夹id',
                                                   "'`member_id`'" STRING COMMENT '会员id',
                                                   "'`supplier_id`'" STRING COMMENT '商家ID',
                                                   "'`add_time`'" STRING COMMENT '添加时间',
                                                   "'`create_time`'" STRING COMMENT '创建时间',
                                                   "'`operate_time`'" STRING COMMENT '操作时间'

) COMMENT '商家收藏事实表，周期性事实表，每日全量加载'
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

supplier_favorites2="
insert overwrite table dwd.dwd_user_member_supplier_favorites_full partition(dt=$dt_casmart_first)
select
    id,
    member_id,
    supplier_id,
    add_time,
    created,
    modified
from ods.ods_sys_tbl_member_favorites_upsert
where dt=$dt_casmart_first and supplier_id IS NOT NULL;
"

cart_create3="
use dwd;
DROP TABLE IF EXISTS dwd_order_member_cart_full;
CREATE TABLE dwd_order_member_cart_full(
                                     "'`cart_id`'" STRING COMMENT '购物车id',
                                     "'`member_id`'" STRING COMMENT '会员id',
                                     "'`product_id`'" STRING COMMENT '商品ID',
                                     "'`supplier_id`'" STRING COMMENT '商家ID',
                                     "'`station_id`'" STRING COMMENT '配送区ID',
                                     "'`order_id`'" STRING COMMENT '订单ID',
                                     "'`product_spu`'" STRING COMMENT '商品spu',
                                     "'`session_id`'" STRING COMMENT 'session_id',
                                     "'`amount`'" bigint COMMENT '加购数量',
                                     "'`price`'" DECIMAL(16,2) COMMENT '商品售价',
                                     "'`add_time`'" STRING COMMENT '下单时间',
                                     "'`create_time`'" STRING COMMENT '创建时间',
                                     "'`operate_time`'" STRING COMMENT '操作时间'
) COMMENT '购物车事实表，周期性事实表，每日全量加载'
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

cart3="
insert overwrite table dwd.dwd_order_member_cart_full partition(dt=$dt_casmart_first)
select
    id,
    buyer_id,
    product_id,
    supplier_id,
    station_id,
    order_id,
    CONCAT(CONCAT(product_brand_id,'_'),product_code),
    session_id,
    amount,
    product_sale_price,
    add_time,
    created,
    modified
from ods.ods_order_tbl_cart_upsert
where dt=$dt_casmart_first ;
"

login_create4="
use dwd;
DROP TABLE IF EXISTS dwd_flow_member_login_result_incr;
CREATE TABLE dwd_flow_member_login_result_incr(
                                             "'`member_id`'" STRING COMMENT '会员id',
                                             "'`account`'" STRING COMMENT '账号名称',
                                             "'`is_login`'" STRING COMMENT '是否登陆',
                                             "'`is_success`'" STRING COMMENT '是否登陆成功',
                                             "'`fail_reason`'" STRING COMMENT '登陆失败原因',
                                             "'`station`'" STRING COMMENT '配送区',
                                             "'`login_time`'" STRING COMMENT '登陆时间'
) COMMENT '会员登陆表，事务型事实表，每日增量加载'
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

#跟神策埋点相关的事务型事实表，首日加载按照月份动态分区，每日加载按天分区
login4="
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
insert overwrite table dwd.dwd_flow_member_login_result_incr partition(dt)
select
    substr(distinct_id,5),
    account,
    is_login,
    case
        when fail_reason='' then '1'
        else '0'
        end,
    fail_reason,
    receive_region,
    "'`time`'",
    dt
FROM ods.ods_sensorslog_events
WHERE dt<=$dt_sensorslog_first and event='LoginResult' and user_type='会员';
"

goods_click_create5="
use dwd;
DROP TABLE IF EXISTS dwd_flow_member_good_click_incr;
CREATE TABLE dwd_flow_member_good_click_incr(
                                           "'`member_id`'" STRING COMMENT '会员id',
                                           "'`product_id`'" STRING COMMENT '商品id',
                                           "'`supplier_id`'" STRING COMMENT '商家id',
                                           "'`station`'" STRING COMMENT '配送区名称',
                                           "'`click_time`'" STRING COMMENT '事件发生时间',
                                           "'`is_login`'" STRING COMMENT '会员是否登陆状态',
                                           "'`is_first_day`'" STRING COMMENT '会员是否当日首次点击此商品',
                                           "'`current_page`'" STRING COMMENT '此商品页面url',
                                           "'`source_page`'" STRING COMMENT '商品所在页面url',
                                           "'`os`'" STRING COMMENT '操作系统',
                                           "'`source_type`'" STRING COMMENT '流量来源'
) COMMENT '商品点击事实表，事务型事实表，每日增量加载'
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

goods_click5="
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
insert overwrite table dwd.dwd_flow_member_good_click_incr partition(dt)
select
    substr(distinct_id,5),
    good_id,
    supplier_id,
    receive_region,
    "'`time`'",
    is_login,
    is_first_day,
    current_page,
    source_page,
    os,
    latest_traffic_source_type,
    dt
FROM ods.ods_sensorslog_events
WHERE dt<=$dt_sensorslog_first and event='goodDetail' and user_type='会员';
"

search_click_create6="
use dwd;
DROP TABLE IF EXISTS dwd_flow_member_search_click_incr;
CREATE TABLE dwd_flow_member_search_click_incr(
                                             "'`member_id`'" STRING COMMENT '会员id',
                                             "'`product_id`'" STRING COMMENT '商品id',
                                             "'`supplier_id`'" STRING COMMENT '商家id',
                                             "'`station`'" STRING COMMENT '配送区名称',
                                             "'`click_time`'" STRING COMMENT '事件发生时间',
                                             "'`is_login`'" STRING COMMENT '会员是否登陆状态',
                                             "'`is_first_day`'" STRING COMMENT '会员是否当日首次点击此商品',
                                             "'`key_word`'" STRING COMMENT '搜索关键词',
                                             "'`position_number`'" STRING COMMENT '点击商品在搜索返回列表的位置序号',
                                             "'`current_page`'" STRING COMMENT '当前页面url',
                                             "'`os`'" STRING COMMENT '操作系统'
) COMMENT '搜索列表点击事实表，事务型事实表，每日增量加载'
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

search_click6="
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
insert overwrite table dwd.dwd_flow_member_search_click_incr partition(dt)
select
    substr(distinct_id,5),
    good_id,
    supplier_id,
    receive_region,
    "'`time`'",
    is_login,
    is_first_day,
    key_word,
    position_number,
    current_page,
    os,
    dt
FROM ods.ods_sensorslog_events
WHERE dt<=$dt_sensorslog_first and event='SearchResultClick' and user_type='会员';
"

orders_goods_create7="
use dwd;
DROP TABLE IF EXISTS dwd_order_orders_goods_zipper;
CREATE TABLE dwd_order_orders_goods_zipper(
                                              "'`orders_goods_id`'" STRING COMMENT '订单详情id',
                                              "'`orders_id`'" STRING COMMENT '订单id',
                                              "'`product_id`'" STRING COMMENT '商品ID',
                                              "'`supplier_id`'" STRING COMMENT '商家ID',
                                              "'`product_spu`'" STRING COMMENT '商品spu',
                                              "'`orders_goods_type`'" STRING COMMENT '订单商品类型',
                                              "'`product_price`'" DECIMAL(16,2) COMMENT '商品价格',
                                              "'`orders_goods_amount`'" bigint COMMENT '购买数量',
                                              "'`orders_received_amount`'" bigint COMMENT '收获数量',
                                              "'`create_time`'" STRING COMMENT '创建时间',
                                              "'`operate_time`'" STRING COMMENT '操作时间',
                                              "'`start_date`'" STRING comment '开始日期',
                                              "'`end_date`'" STRING comment '结束日期'
) COMMENT '订单详情事实表，周期型快照事实表,拉链表'
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

orders_goods7="
insert overwrite table dwd.dwd_order_orders_goods_zipper partition(dt='99999999')
select
    id,
    orders_id,
    product_id,
    product_supplier_id,
    CONCAT(CONCAT(product_brand_id, '_'), product_code),
    "'`type`'",
    product_price,
    amount,
    received_amount,
    created,
    modified,
    $dt_casmart_first,
    '99999999'
from ods.ods_order_tbl_orders_goods_upsert WHERE dt=$dt_casmart_first;
"

order_create8="
use dwd;
DROP TABLE IF EXISTS dwd_order_orders_incr;
CREATE TABLE dwd_order_orders_incr (
                                       "'`id`'" STRING COMMENT '订单id',
                                       "'`member_id`'" STRING COMMENT '购买人id，会员id',
                                       "'`institutes_id`'" STRING COMMENT '研究所id',
                                       "'`supplier_id`'" STRING COMMENT '商家id',
                                       "'`station_id`'" STRING COMMENT '配送区id',
                                       "'`address_id`'" STRING COMMENT '地址id',
                                       "'`delivery_time_id`'" STRING COMMENT '配送周期id',
                                       "'`total_mkt_price`'" STRING COMMENT '商品总市场价',
                                       "'`total_price`'" STRING COMMENT '商品总价格',
                                       "'`total_freight`'" STRING COMMENT '订单总运费',
                                       "'`total_all_price`'" STRING COMMENT '订单总价格',
                                       "'`order_type`'" STRING COMMENT '订单类型',
                                       "'`order_sn`'" STRING COMMENT '订单号',
                                       "'`pay_way`'" STRING COMMENT '付款方式',
                                       "'`pay_way_name`'" STRING COMMENT '付款方式名称',
                                       "'`pay_type`'" STRING COMMENT '付款类型',
                                       "'`pay_type_name`'" STRING COMMENT '付款类型名称',
                                       "'`payment_status`'" STRING COMMENT '支付状态',
                                       "'`payment_status_time`'" STRING COMMENT '支付时间',
                                       "'`delivery`'" STRING COMMENT '配送方式',
                                       "'`delivery_name`'" STRING COMMENT '配送方式名称',
                                       "'`delivery_status`'" STRING COMMENT '配送状态',
                                       "'`delivery_status_time`'" STRING COMMENT '配送时间',
                                       "'`audit_status`'" STRING COMMENT '主账户审核状态',
                                       "'`flow_audit_status`'" STRING COMMENT '工作流审核状态',
                                       "'`fail_note`'" STRING COMMENT '订单失败日志',
                                       "'`created_time`'" STRING COMMENT '订单创建时间',
                                       "'`unconfirmed_time`'" STRING COMMENT '订单未确认时间',
                                       "'`confirmed_time`'" STRING COMMENT '订单确认时间',
                                       "'`succeed_time`'" STRING COMMENT '订单成功时间',
                                       "'`failed_time`'" STRING COMMENT '订单失败时间'
) COMMENT '订单事实表，累积型快照事实表'
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

# 累积型快照事实表,要将修改时间增加成各个状态0123的时间字段,9999分区存放的是状态还会变化的订单数据，其他分区存放的是状态不会变化的订单数据
# 首日加载动态分区，状态不会变化的订单数据按照最终状态的时间动态分区，状态还会变化的订单数据放在9999分区
# 因为集群资源不足，此累积型快照事实表是按月分区的，首日和每日都是
order8="
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
with
    tbl_time0 as (SELECT id,modified FROM ods.ods_order_tbl_orders_upsert WHERE dt='20220218' and status=0),
    tbl_time1 as (SELECT id,modified FROM ods.ods_order_tbl_orders_upsert WHERE dt='20220218' and status=1),
    tbl_time2 as (SELECT id,modified FROM ods.ods_order_tbl_orders_upsert WHERE dt='20220218' and status=2),
    tbl_time3 as (SELECT id,modified FROM ods.ods_order_tbl_orders_upsert WHERE dt='20220218' and status=3),
    tbl_time as (SELECT nvl(nvl(nvl(tbl_time0.id,tbl_time1.id),tbl_time2.id),tbl_time3.id) id,
                        tbl_time0.modified unconfirmed_time,
                        tbl_time1.modified confirmed_time,
                        tbl_time2.modified succeed_time,
                        tbl_time3.modified failed_time
                from tbl_time0
                full outer join tbl_time1 on tbl_time0.id= tbl_time1.id
                full outer join tbl_time2 on tbl_time0.id= tbl_time2.id
                full outer join tbl_time3 on tbl_time0.id= tbl_time3.id)
insert overwrite table dwd.dwd_order_orders_incr partition(dt)
select
    tbl_order.id,
    buyer_id,
    institutes,
    supplier_id,
    station_id,
    address_id,
    delivery_time_id,
    total_mkt_price,
    total_price,
    total_freight,
    total_all_price,
    "'`type`'",
    sn,
    pay_way,
    pay_way_name,
    pay_type,
    pay_type_name,
    payment_status,
    payment_status_time,
    delivery,
    delivery_name,
    delivery_status,
    delivery_status_time,
    audit_status,
    flow_audit_status,
    fail_note,
    created,
    unconfirmed_time,
    confirmed_time,
    succeed_time,
    failed_time,
    CASE
        when failed_time is not null then CONCAT(substr(cast(date_format(failed_time,'yyyyMMdd') as string),0,6),'01')
        when succeed_time is not null then CONCAT(substr(cast(date_format(succeed_time,'yyyyMMdd') as string),0,6),'01')
        when confirmed_time is not null and translate(CAST(date_add(date_format(confirmed_time,'yyyy-MM-dd'),365) as string),'-','')<=$dt_casmart_first then CONCAT(substr(cast(date_format(confirmed_time,'yyyyMMdd') as string),0,6),'01')
        when unconfirmed_time is not null and translate(CAST(date_add(date_format(unconfirmed_time,'yyyy-MM-dd'),365) as string),'-','')<=$dt_casmart_first then CONCAT(substr(cast(date_format(unconfirmed_time,'yyyyMMdd') as string),0,6),'01')
        else '99999999'
    END
from (select * from ods.ods_order_tbl_orders_upsert where dt=$dt_casmart_first) tbl_order
left join tbl_time
on tbl_order.id=tbl_time.id;
"


case $1 in
    product_favorites )
        hive -e "$product_favorites_create1"
        hive -e "$product_favorites1"
    ;;
    supplier_favorites )
        hive -e "$supplier_favorites_create2"
        hive -e "$supplier_favorites2"
    ;;
    cart )
        hive -e "$cart_create3"
        hive -e "$cart3"
    ;;
    login4 )
        hive -e "$login_create4"
        hive -e "$login4"
    ;;
    goods_click )
        hive -e "$goods_click_create5"
        hive -e "$goods_click5"
    ;;
    search_click )
        hive -e "$search_click_create6"
        hive -e "$search_click6"
    ;;
    orders_goods )
        hive -e "$orders_goods_create7"
        hive -e "$orders_goods7"
    ;;
    order )
        hive -e "$order_create8"
        hive -e "$order8"
    ;;
    all )
        hive -e "$product_favorites_create1"
        hive -e "$product_favorites1"
        hive -e "$supplier_favorites_create2"
        hive -e "$supplier_favorites2"
        hive -e "$cart_create3"
        hive -e "$cart3"
        hive -e "$login_create4"
        hive -e "$login4"
        hive -e "$goods_click_create5"
        hive -e "$goods_click5"
        hive -e "$search_click_create6"
        hive -e "$search_click6"
        hive -e "$orders_goods_create7"
        hive -e "$orders_goods7"
        hive -e "$order_create8"
        hive -e "$order8"
    ;;
esac





















