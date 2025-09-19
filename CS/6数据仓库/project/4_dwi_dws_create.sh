#!/bin/bash

member_create="
DROP TABLE IF EXISTS dws.dws_user_member_wide;
CREATE TABLE dws.dws_user_member_wide(
                                                  "'`member_id`'" STRING  COMMENT '会员id',
                                                  "'`cart_1d_count`'" bigint  COMMENT '近1天加购次数',
                                                  "'`cart_7d_count`'" bigint  COMMENT '近7天加购次数',
                                                  "'`click_1d_count`'" bigint  COMMENT '近1天点击次数',
                                                  "'`click_7d_count`'" bigint  COMMENT '近7天点击次数',
                                                  "'`click_30d_count`'" bigint  COMMENT '近30天点击次数',
                                                  "'`click_total_count`'" bigint  COMMENT '累计点击次数',
                                                  "'`login_1d_count`'" bigint  COMMENT '近1天登录次数',
                                                  "'`login_30d_count`'" bigint  COMMENT '近30天登录次数',
                                                  "'`login_30d_days`'" bigint  COMMENT '近30天登录天数',
                                                  "'`login_7d_days`'" bigint  COMMENT '近7天登录天数',
                                                  "'`login_total_days`'" bigint  COMMENT '累计登录天数',
                                                  "'`login_last_date`'" string  COMMENT '最后登陆日期',
                                                  "'`favorites_1d_count`'" bigint  COMMENT '近1天收藏次数',
                                                  "'`favorites_7d_count`'" bigint  COMMENT '近7天收藏次数',
                                                  "'`search_1d_count`'" bigint  COMMENT '近1天搜索次数',
                                                  "'`search_7d_count`'" bigint  COMMENT '近7天搜索次数',
                                                  "'`search_30d_count`'" bigint  COMMENT '近30天搜索次数',
                                                  "'`orders_1d_count`'" bigint  COMMENT '近1天下单成功次数',
                                                  "'`orders_7d_count`'" bigint  COMMENT '近7天下单成功次数',
                                                  "'`orders_30d_count`'" bigint  COMMENT '近30天下单成功次数',
                                                  "'`orders_30d_days`'" bigint  COMMENT '近30天下单成功天数',
                                                  "'`orders_last_date`'" string  COMMENT '最后下单成功日期',
                                                  "'`orders_1d_price`'" DECIMAL(16,2)  COMMENT '近1天订单商品价格',
                                                  "'`orders_7d_price`'" DECIMAL(16,2)  COMMENT '近7天订单商品价格',
                                                  "'`orders_30d_price`'" DECIMAL(16,2)  COMMENT '近30天订单商品价格',
                                                  "'`orders_1d_all_price`'" DECIMAL(16,2)  COMMENT '近1天订单总价格',
                                                  "'`orders_7d_all_price`'" DECIMAL(16,2)  COMMENT '近7天订单总价格',
                                                  "'`orders_30d_all_price`'" DECIMAL(16,2)  COMMENT '近30天订单总价格'
)COMMENT '会员主题聚合表'
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"
hive -e "$member_create"

supplier_create="
DROP TABLE IF EXISTS dws.dws_user_supplier_wide;
CREATE TABLE dws.dws_user_supplier_wide(
                                                    "'`supplier_id`'" STRING  COMMENT '商家id',
                                                    "'`cart_1d_count`'" bigint  COMMENT '近1天加购次数',
                                                    "'`cart_7d_count`'" bigint  COMMENT '近7天加购次数',
                                                    "'`click_1d_count`'" bigint  COMMENT '近1天点击次数',
                                                    "'`click_7d_count`'" bigint  COMMENT '近7天点击次数',
                                                    "'`click_30d_count`'" bigint  COMMENT '近30天点击次数',
                                                    "'`click_total_count`'" bigint  COMMENT '累计点击次数',
                                                    "'`favorites_1d_count`'" bigint  COMMENT '近1天收藏次数',
                                                    "'`favorites_7d_count`'" bigint  COMMENT '近7天收藏次数',
                                                    "'`search_1d_count`'" bigint  COMMENT '近1天搜索次数',
                                                    "'`search_7d_count`'" bigint  COMMENT '近7天搜索次数',
                                                    "'`search_30d_count`'" bigint  COMMENT '近30天搜索次数',
                                                    "'`orders_1d_count`'" bigint  COMMENT '近1天下单成功次数',
                                                    "'`orders_7d_count`'" bigint  COMMENT '近7天下单成功次数',
                                                    "'`orders_30d_count`'" bigint  COMMENT '近30天下单成功次数',
                                                    "'`orders_30d_days`'" bigint  COMMENT '近30天下单成功天数',
                                                    "'`orders_last_date`'" string  COMMENT '最后下单成功日期',
                                                    "'`orders_1d_price`'" DECIMAL(16,2)  COMMENT '近1天订单商品价格',
                                                    "'`orders_7d_price`'" DECIMAL(16,2)  COMMENT '近7天订单商品价格',
                                                    "'`orders_30d_price`'" DECIMAL(16,2)  COMMENT '近30天订单商品价格',
                                                    "'`orders_1d_all_price`'" DECIMAL(16,2)  COMMENT '近1天订单总价格',
                                                    "'`orders_7d_all_price`'" DECIMAL(16,2)  COMMENT '近7天订单总价格',
                                                    "'`orders_30d_all_price`'" DECIMAL(16,2)  COMMENT '近30天订单总价格'
)COMMENT '商家主题聚合表'
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"
hive -e "$supplier_create"

product_create="
DROP TABLE IF EXISTS dws.dws_product_product_wide;
CREATE TABLE dws.dws_product_product_wide(
                                                      "'`product_id`'" STRING  COMMENT '商品id',
                                                      "'`product_spu`'" STRING  COMMENT '商品spu',
                                                      "'`cart_1d_count`'" bigint  COMMENT '近1天加购次数',
                                                      "'`cart_7d_count`'" bigint  COMMENT '近7天加购次数',
                                                      "'`cart_1d_amount`'" bigint  COMMENT '近1天加购数量',
                                                      "'`cart_7d_amount`'" bigint  COMMENT '近7天加购数量',
                                                      "'`favorites_1d_count`'" bigint  COMMENT '近1天收藏次数',
                                                      "'`favorites_7d_count`'" bigint  COMMENT '近7天收藏次数',
                                                      "'`click_1d_count`'" bigint  COMMENT '近1天点击次数',
                                                      "'`click_7d_count`'" bigint  COMMENT '近7天点击次数',
                                                      "'`click_30d_count`'" bigint  COMMENT '近30天点击次数',
                                                      "'`click_total_count`'" bigint  COMMENT '累计点击次数',
                                                      "'`search_1d_count`'" bigint  COMMENT '近1天搜索次数',
                                                      "'`search_7d_count`'" bigint  COMMENT '近7天搜索次数',
                                                      "'`search_30d_count`'" bigint  COMMENT '近30天搜索次数',
                                                      "'`orders_1d_count`'" bigint  COMMENT '近1天下单成功次数',
                                                      "'`orders_7d_count`'" bigint  COMMENT '近7天下单成功次数',
                                                      "'`orders_30d_count`'" bigint  COMMENT '近30天下单成功次数',
                                                      "'`orders_30d_days`'" bigint  COMMENT '近30天下单成功天数',
                                                      "'`orders_last_date`'" string  COMMENT '最后下单成功日期',
                                                      "'`orders_1d_amount`'" bigint  COMMENT '近1天下单成功数量',
                                                      "'`orders_7d_amount`'" bigint  COMMENT '近7天下单成功数量',
                                                      "'`orders_30d_amount`'" bigint  COMMENT '近30天下单成功数量'
)COMMENT '商品主题聚合表'
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"
hive -e "$product_create"



























































