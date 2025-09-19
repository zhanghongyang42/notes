#!/bin/bash


do_date='20220218'



#-----------------------------------------------------------------------------------------------------------------------


member_create="
DROP TABLE IF EXISTS dwi.dwi_user_member_day;
CREATE TABLE dwi.dwi_user_member_day(
                                        "'`member_id`'" STRING COMMENT '会员id',
                                        "'`cart_count`'" bigint COMMENT '每天加购次数',
                                        "'`cart_price`'" DECIMAL(16,2) COMMENT '每天加购商品总价',
                                        "'`click_count`'" bigint COMMENT '每天点击次数',
                                        "'`click_first_count`'" bigint COMMENT '每天第一次点击商品次数',
                                        "'`login_count`'" bigint COMMENT '每天登录次数',
                                        "'`favorites_count`'" bigint COMMENT '会员每日收藏商品次数',
                                        "'`search_count`'" bigint COMMENT '会员每天搜索次数',
                                        "'`orders_count`'" bigint COMMENT '会员每天下单次数',
                                        "'`orders_price`'" DECIMAL(16,2) COMMENT '订单商品价格',
                                        "'`orders_all_price`'" DECIMAL(16,2) COMMENT '订单总价格'
) COMMENT '每日会员行为'
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

supplier_create="
DROP TABLE IF EXISTS dwi.dwi_user_supplier_day;
CREATE TABLE dwi.dwi_user_supplier_day(
                                          "'`supplier_id`'" STRING COMMENT '商家id',
                                          "'`cart_count`'" bigint COMMENT '加购次数',
                                          "'`cart_price`'" DECIMAL(16,2) COMMENT '商家每日加购商品总价',
                                          "'`click_count`'" bigint COMMENT '点击次数',
                                          "'`click_Search_count`'" bigint COMMENT '搜索次数',
                                          "'`favorites_count`'" bigint COMMENT '商家被收藏次数',
                                          "'`orders_count`'" bigint COMMENT '下单次数',
                                          "'`orders_price`'" DECIMAL(16,2) COMMENT '订单实际价格',
                                          "'`orders_all_price`'" DECIMAL(16,2) COMMENT '订单总价格'
) COMMENT '每日商家行为'
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

product_create="
DROP TABLE IF EXISTS dwi.dwi_product_product_day;
CREATE TABLE dwi.dwi_product_product_day(
                                            "'`product_id`'" STRING COMMENT '商品id',
                                            "'`product_spu`'" STRING COMMENT '商品spu',
                                            "'`cart_count`'" bigint COMMENT '被加购次数',
                                            "'`cart_amount`'" bigint COMMENT '被加购数量',
                                            "'`favorites_count`'" bigint COMMENT '被收藏次数',
                                            "'`click_count`'" bigint COMMENT '被点击次数',
                                            "'`click_Search_count`'" bigint COMMENT '被搜索次数',
                                            "'`orders_goods_count`'" bigint COMMENT '被下单次数',
                                            "'`orders_goods_amount`'" bigint COMMENT '被下单数量'
) COMMENT '每日商品属性'
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

member_first="
with
    cart_tmp as (SELECT
                     member_id,
                     count(*) cart_count,
                     sum(price) cart_price
                 FROM dwd.dwd_order_member_cart_full
                 WHERE member_id is not null and dt=$do_date
                 GROUP by member_id),
    click_tmp as (SELECT
                      member_id,
                      COUNT(*) click_count,
                      SUM(is_first_day) click_first_count
                  FROM dwd.dwd_flow_member_good_click_incr
                  where member_id is not null and dt=$do_date
                  GROUP BY member_id),
    login_tmp as (SELECT
                      member_id,
                      COUNT(*) login_count
                  FROM dwd.dwd_flow_member_login_result_incr
                  WHERE member_id is not null and dt='$do_date'
                  GROUP by member_id),
    favor_tmp as (SELECT
                      member_id,
                      COUNT(*) favorites_count
                  FROM dwd.dwd_user_member_product_favorites_full
                  WHERE member_id is not null and dt='$do_date'
                  GROUP by member_id),
    search_tmp as (SELECT
                       member_id,
                       COUNT(*) search_count
                   FROM dwd.dwd_flow_member_search_click_incr
                   WHERE member_id is not null and dt='$do_date'
                   GROUP BY member_id),
    orders_tmp as (SELECT
                       member_id,
                       COUNT(*) orders_count,
                       SUM(total_price) orders_price,
                       SUM(total_all_price) orders_all_price
                   FROM dwd.dwd_order_orders_incr
                   WHERE regexp_replace(SUBSTR(succeed_time,0,10),'-','')='$do_date'
                   GROUP BY member_id),
    member_9999_tmp as (SELECT member_id FROM dim.dim_user_member_zipper WHERE dt='99999999')
insert overwrite table dwi.dwi_user_member_day partition(dt='$do_date')
select
    member_9999_tmp.member_id,
    cart_count,
    cart_price,
    click_count,
    click_first_count,
    login_count,
    favorites_count,
    search_count,
    orders_count,
    orders_price,
    orders_all_price
FROM member_9999_tmp
         left join cart_tmp
                   on member_9999_tmp.member_id = cart_tmp.member_id
         LEFT JOIN click_tmp
                   on member_9999_tmp.member_id = click_tmp.member_id
         LEFT JOIN login_tmp
                   on member_9999_tmp.member_id = login_tmp.member_id
         LEFT JOIN favor_tmp
                   on member_9999_tmp.member_id = favor_tmp.member_id
         LEFT JOIN search_tmp
                   on member_9999_tmp.member_id = search_tmp.member_id
         LEFT JOIN orders_tmp
                   on member_9999_tmp.member_id = orders_tmp.member_id
WHERE cart_count is not null or cart_price is not null or click_count is not null or click_first_count is not null or login_count is not null
        or favorites_count is not null or search_count is not null or orders_count is not null or orders_price is not null or orders_all_price is not null;
"

supplier_first="
with
    cart_tmp as (SELECT
                     supplier_id,
                     count(*) cart_count,
                     sum(price) cart_price
                 FROM dwd.dwd_order_member_cart_full WHERE dt='$do_date' GROUP by supplier_id),
    click_tmp as (SELECT
                      supplier_id,
                      COUNT(*) click_count
                  FROM dwd.dwd_flow_member_good_click_incr
                  where supplier_id is not null and dt='$do_date'
                  GROUP BY supplier_id),
    search_tmp as (SELECT
                       supplier_id,
                       COUNT(*) search_count
                   FROM dwd.dwd_flow_member_search_click_incr
                   WHERE supplier_id is not null and dt='$do_date'
                   GROUP BY supplier_id),
    favor_tmp as (SELECT
                      supplier_id,
                      COUNT(*) favorites_count
                  FROM dwd.dwd_user_member_supplier_favorites_full
                  WHERE  dt='$do_date'
                  GROUP by supplier_id),
    orders_tmp as (SELECT
                       supplier_id,
                       COUNT(*) orders_count,
                       SUM(total_price) orders_price,
                       SUM(total_all_price) orders_all_price
                   FROM dwd.dwd_order_orders_incr
                   WHERE regexp_replace(SUBSTR(succeed_time,0,10),'-','')='$do_date'
                   GROUP BY supplier_id),
    supplier_9999_tmp as (SELECT * FROM dim.dim_user_supplier_zipper WHERE dt='99999999')
insert overwrite table dwi.dwi_user_supplier_day partition(dt='$do_date')
select
    supplier_9999_tmp.supplier_id,
    cart_count,
    cart_price,
    click_count,
    search_count,
    favorites_count,
    orders_count,
    orders_price,
    orders_all_price
FROM supplier_9999_tmp
         left join cart_tmp
                   on supplier_9999_tmp.supplier_id = cart_tmp.supplier_id
         LEFT JOIN click_tmp
                   on supplier_9999_tmp.supplier_id = click_tmp.supplier_id
         LEFT JOIN favor_tmp
                   on supplier_9999_tmp.supplier_id = favor_tmp.supplier_id
         LEFT JOIN search_tmp
                   on supplier_9999_tmp.supplier_id = search_tmp.supplier_id
         LEFT JOIN orders_tmp
                   on supplier_9999_tmp.supplier_id = orders_tmp.supplier_id
where cart_count is not null or cart_price is not null or click_count is not null
   or search_count is not null
   or favorites_count is not null
   or orders_count is not null
   or orders_price is not null
   or orders_all_price is not null;
"

product_first="
with
    cart_tmp as (SELECT
                     product_id,
                     count(*) cart_count,
                     SUM(amount) cart_amount
                 FROM dwd.dwd_order_member_cart_full
                 WHERE dt='$do_date'
                 GROUP BY product_id),
    favor_tmp as (SELECT
                      product_id,
                      COUNT(*) favorites_count
                  FROM dwd.dwd_user_member_product_favorites_full
                  WHERE  dt='$do_date'
                  GROUP by product_id),
    click_tmp as (SELECT
                      product_id,
                      COUNT(*) click_count
                  FROM dwd.dwd_flow_member_good_click_incr
                  where dt='$do_date'
                  GROUP BY product_id),
    search_tmp as (SELECT
                       product_id,
                       COUNT(*) search_count
                   FROM dwd.dwd_flow_member_search_click_incr
                   WHERE dt='$do_date'
                   GROUP BY product_id),
    orders_goods_tmp as (SELECT
                             product_id,
                             count(*) orders_goods_count,
                             sum(orders_goods_amount) orders_goods_amount
                         FROM dwd.dwd_order_orders_goods_zipper
                         WHERE dt='99999999' and regexp_replace(SUBSTR(operate_time,0,10),'-','')='$do_date'
                         group by product_id),
    product_9999_tmp as (SELECT * FROM dim.dim_product_product_zipper WHERE dt='99999999')
insert overwrite table dwi.dwi_product_product_day partition(dt='$do_date')
select
    product_9999_tmp.product_id,
    product_9999_tmp.product_spu,
    cart_count,
    cart_amount,
    favorites_count,
    click_count,
    search_count,
    orders_goods_count,
    orders_goods_amount
FROM product_9999_tmp
         left join cart_tmp
                   on product_9999_tmp.product_id = cart_tmp.product_id
         LEFT JOIN favor_tmp
                   on product_9999_tmp.product_id = favor_tmp.product_id
         LEFT JOIN click_tmp
                   on product_9999_tmp.product_id = click_tmp.product_id
         LEFT JOIN search_tmp
                   on product_9999_tmp.product_id = search_tmp.product_id
         LEFT JOIN orders_goods_tmp
                   on product_9999_tmp.product_id = orders_goods_tmp.product_id
where cart_count is not null or cart_amount is not null or favorites_count is not null
   or click_count is not null or search_count is not null or orders_goods_count is not null or orders_goods_amount is not null;
"




hive -e "$member_create"
hive -e "$supplier_create"
hive -e "$product_create"
hive -e "$member_first"
hive -e "$supplier_first"
hive -e "$product_first"























