#!/bin/bash
#ods开始日期到今天的每日加载 dwd层数据得补上


#获取输入日期2022-02-02 或者 当天日期的前一天日期
if [ -n "$2" ] ;then
    do_dateg=$2
    do_date=${do_dateg//-/}
else
    do_dateg=`date -d '-1 day' +%F`
    do_date=${do_dateg//-/}
fi

product_favorites1="
insert overwrite table dwd.dwd_user_member_product_favorites_full partition(dt=$do_date)
select
    id,
    member_id,
    target_id,
    add_time,
    created,
    modified
from ods.ods_sys_tbl_member_favorites_upsert
where dt=$do_date;
"

supplier_favorites2="
insert overwrite table dwd.dwd_user_member_supplier_favorites_full partition(dt=$do_date)
select
    id,
    member_id,
    supplier_id,
    add_time,
    created,
    modified
from ods.ods_sys_tbl_member_favorites_upsert
where dt=$do_date and supplier_id IS NOT NULL ;
"

cart3="
insert overwrite table dwd.dwd_order_member_cart_full partition(dt=$do_date)
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
where dt=$do_date ;
"

login4="
insert overwrite table dwd.dwd_flow_member_login_result_incr partition(dt=$do_date)
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
    "'`time`'"
FROM ods.ods_sensorslog_events
WHERE dt=$do_date and event='LoginResult' and user_type='会员';
"

good_click5="
insert overwrite table dwd.dwd_flow_member_good_click_incr partition(dt=$do_date)
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
    latest_traffic_source_type
FROM ods.ods_sensorslog_events
WHERE dt=$do_date and event='goodDetail' and user_type='会员';
"

search_click6="
insert overwrite table dwd.dwd_flow_member_search_click_incr partition(dt=$do_date)
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
    os
FROM ods.ods_sensorslog_events
WHERE dt=$do_date and event='SearchResultClick' and user_type='会员';
"

orders_goods7="
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
with
    dwd_order_orders_goods_zipper_9999 as (select * from dwd.dwd_order_orders_goods_zipper where dt = '99999999'),
    ods_new as(SELECT
                   id,
                   orders_id,
                   product_id,
                   product_supplier_id,
                   CONCAT(CONCAT(product_brand_id, '_'), product_code) cc,
                   "'`type`'",
                   product_price,
                   amount,
                   received_amount,
                   created,
                   modified,
                   $do_date aa,
                   '99999999' bb
               FROM ods.ods_order_tbl_orders_goods_upsert
               WHERE dt=$do_date),
    tmp as (select
                olda.*,
                newa.id opid,
                newa.orders_id oid,
                newa.product_id pid,
                newa.product_supplier_id sid,
                newa.cc pspu,
                newa."'`type`'" otype,
                newa.product_price ppri,
                newa.amount amo,
                newa.received_amount ramo,
                newa.created cre,
                newa.modified moi,
                newa.aa sta,
                newa.bb endt
            from dwd_order_orders_goods_zipper_9999 olda
            full outer join ods_new newa
            on olda.orders_goods_id = newa.id)
insert overwrite table dwd.dwd_order_orders_goods_zipper partition(dt)
select
    nvl(opid, orders_goods_id),
    nvl(oid, orders_id),
    nvl(pid, product_id),
    nvl(sid, supplier_id),
    nvl(pspu, product_spu),
    nvl(otype, orders_goods_type),
    nvl(ppri, product_price),
    nvl(amo, orders_goods_amount),
    nvl(ramo, orders_received_amount),
    nvl(cre, create_time),
    nvl(moi, operate_time),
    nvl(sta, start_date),
    nvl(endt, end_date),
    nvl(endt, end_date) dt
from tmp
union all
select
    orders_goods_id,
    orders_id,
    product_id,
    supplier_id,
    product_spu,
    orders_goods_type,
    product_price,
    orders_goods_amount,
    orders_received_amount,
    create_time,
    operate_time,
    start_date,
    regexp_replace(cast(date_add('$do_dateg', -1) as string), '-', ''),
    regexp_replace(cast(date_add('$do_dateg', -1) as string), '-', '') dt
from tmp
where orders_goods_id is not null and opid is not null;
"

# 因为时效性问题，如果订单在一天内就完成了多个状态，数仓中只会记录这天的最后一个状态的时间
# 因为是按月分区的，所以不可以覆盖写入，否则会覆盖之前月份的数据，这里追加写入，但是9999分区数据必须覆盖写入，使用9900完成覆盖写入
orders8="
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
with
    dwd_order_orders_incr_9999 as (select * from dwd.dwd_order_orders_incr where dt = '99999999'),
    ods_tbl_time0 as (SELECT id, modified FROM ods.ods_order_tbl_orders_upsert WHERE dt = $do_date and status = 0),
    ods_tbl_time1 as (SELECT id, modified FROM ods.ods_order_tbl_orders_upsert WHERE dt = $do_date and status = 1),
    ods_tbl_time2 as (SELECT id, modified FROM ods.ods_order_tbl_orders_upsert WHERE dt = $do_date and status = 2),
    ods_tbl_time3 as (SELECT id, modified FROM ods.ods_order_tbl_orders_upsert WHERE dt = $do_date and status = 3),
    ods_tbl_time as (SELECT nvl(nvl(nvl(ods_tbl_time0.id, ods_tbl_time1.id), ods_tbl_time2.id), ods_tbl_time3.id) id,
                             ods_tbl_time0.modified                                                                unconfirmed_time,
                             ods_tbl_time1.modified                                                                confirmed_time,
                             ods_tbl_time2.modified                                                                succeed_time,
                             ods_tbl_time3.modified                                                                failed_time
                    from ods_tbl_time0
                    full outer join ods_tbl_time1 on ods_tbl_time0.id = ods_tbl_time1.id
                    full outer join ods_tbl_time2 on ods_tbl_time0.id = ods_tbl_time2.id
                    full outer join ods_tbl_time3 on ods_tbl_time0.id = ods_tbl_time3.id),
    ods_new as (select tbl_order.id,
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
                        failed_time
                 from (select * from ods.ods_order_tbl_orders_upsert where dt=$do_date) tbl_order
                 left join ods_tbl_time on tbl_order.id = ods_tbl_time.id),
    dwd_new as (select nvl(ods_new.id, dwd_order_orders_incr_9999.id),
                        nvl(ods_new.buyer_id, dwd_order_orders_incr_9999.member_id),
                        nvl(ods_new.institutes, dwd_order_orders_incr_9999.institutes_id),
                        nvl(ods_new.supplier_id, dwd_order_orders_incr_9999.supplier_id),
                        nvl(ods_new.station_id, dwd_order_orders_incr_9999.station_id),
                        nvl(ods_new.address_id, dwd_order_orders_incr_9999.address_id),
                        nvl(ods_new.delivery_time_id, dwd_order_orders_incr_9999.delivery_time_id),
                        nvl(ods_new.total_mkt_price, dwd_order_orders_incr_9999.total_mkt_price),
                        nvl(ods_new.total_price, dwd_order_orders_incr_9999.total_price),
                        nvl(ods_new.total_freight, dwd_order_orders_incr_9999.total_freight),
                        nvl(ods_new.total_all_price, dwd_order_orders_incr_9999.total_all_price),
                        nvl(ods_new."'`type`'", dwd_order_orders_incr_9999.order_type),
                        nvl(ods_new.sn, dwd_order_orders_incr_9999.order_sn),
                        nvl(ods_new.pay_way, dwd_order_orders_incr_9999.pay_way),
                        nvl(ods_new.pay_way_name, dwd_order_orders_incr_9999.pay_way_name),
                        nvl(ods_new.pay_type, dwd_order_orders_incr_9999.pay_type),
                        nvl(ods_new.pay_type_name, dwd_order_orders_incr_9999.pay_type_name),
                        nvl(ods_new.payment_status, dwd_order_orders_incr_9999.payment_status),
                        nvl(ods_new.payment_status_time, dwd_order_orders_incr_9999.payment_status_time),
                        nvl(ods_new.delivery, dwd_order_orders_incr_9999.delivery),
                        nvl(ods_new.delivery_name, dwd_order_orders_incr_9999.delivery_name),
                        nvl(ods_new.delivery_status, dwd_order_orders_incr_9999.delivery_status),
                        nvl(ods_new.delivery_status_time, dwd_order_orders_incr_9999.delivery_status_time),
                        nvl(ods_new.audit_status, dwd_order_orders_incr_9999.audit_status),
                        nvl(ods_new.flow_audit_status, dwd_order_orders_incr_9999.flow_audit_status),
                        nvl(ods_new.fail_note, dwd_order_orders_incr_9999.fail_note),
                        nvl(ods_new.created, dwd_order_orders_incr_9999.created_time),
                        nvl(ods_new.unconfirmed_time, dwd_order_orders_incr_9999.unconfirmed_time) unconfirmed_time,
                        nvl(ods_new.confirmed_time, dwd_order_orders_incr_9999.confirmed_time)     confirmed_time,
                        nvl(ods_new.succeed_time, dwd_order_orders_incr_9999.succeed_time)         succeed_time,
                        nvl(ods_new.failed_time, dwd_order_orders_incr_9999.failed_time)           failed_time
                 from dwd_order_orders_incr_9999
                 full outer join ods_new
                 on dwd_order_orders_incr_9999.id = ods_new.id)
insert into table dwd.dwd_order_orders_incr partition (dt)
select *,
       CASE
           when failed_time is not null then CONCAT(substr(cast(date_format(failed_time, 'yyyyMMdd') as string), 0, 6),'01')
           when succeed_time is not null then CONCAT(substr(cast(date_format(succeed_time, 'yyyyMMdd') as string), 0, 6), '01')
           when confirmed_time is not null and succeed_time is null and failed_time is null and translate(CAST(date_add(date_format(confirmed_time, 'yyyy-MM-dd'), 365) as string), '-', '') <='20220218' then CONCAT(substr(cast(date_format(confirmed_time, 'yyyyMMdd') as string), 0, 6), '01')
           when unconfirmed_time is not null and succeed_time is null and failed_time is null and confirmed_time is null and translate(CAST(date_add(date_format(unconfirmed_time, 'yyyy-MM-dd'), 365) as string), '-', '') <='20220218' then CONCAT(substr(cast(date_format(unconfirmed_time, 'yyyyMMdd') as string), 0, 6), '01')
           else '99990000'
        END
from dwd_new;
insert overwrite table dwd.dwd_order_orders_incr partition (dt='99999999')
select id,
       member_id,
       institutes_id,
       supplier_id,
       station_id,
       address_id,
       delivery_time_id,
       total_mkt_price,
       total_price,
       total_freight,
       total_all_price,
       order_type,
       order_sn,
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
       created_time,
       unconfirmed_time,
       confirmed_time,
       succeed_time,
       failed_time
FROM dwd.dwd_order_orders_incr
WHERE dt = '99990000';
"

# 假设某累积型快照事实表，某天所有的业务记录全部完成，则会导致9999-99-99分区的数据未被覆盖，从而导致数据重复，该函数根据9999-99-99分区的数据的末次修改时间判断其是否被覆盖了，如果未被覆盖，就手动清理
clear_data(){
    #当前日期
    current_date=`date +%F`
    current_date_timestamp=`date -d "$current_date" +%s`

    #9999分区文件夹最近一次修改日期
    last_modified_date=`hadoop fs -ls /user/hive/warehouse/dwd.db/dwd_order_orders_incr | grep '99999999' | awk '{print $6}'`  #2022-03-04
    last_modified_date_timestamp=`date -d "$last_modified_date" +%s`

    #如果最后一次9999文件夹日期小于当前日期，说明今天9999分区没有更新
    if [[ $last_modified_date_timestamp -lt $current_date_timestamp ]]; then
        echo "clear table dwd_order_orders_incr partition(dt=99999999)"
        hadoop fs -rm -r -f /user/hive/warehouse/dwd.db/dwd_order_orders_incr/dt=99999999/*
    fi
}

case $1 in
    product_favorites )
        hive -e "$product_favorites1"
    ;;
    supplier_favorites )
        hive -e "$supplier_favorites2"
    ;;
    cart )
        hive -e "$cart3"
    ;;
    login )
        hive -e "$login4"
    ;;
    good_click )
        hive -e "$good_click5"
    ;;
    search_click )
        hive -e "$search_click6"
    ;;
    orders_goods )
        hive -e "$orders_goods7"
    ;;
    orders )
        hive -e "$orders8"
        hive -e "alter table dwd.dwd_order_orders_incr drop partition (dt = '99990000')"
        clear_data
    ;;
    all )
        hive -e "$product_favorites1"
        hive -e "$supplier_favorites2"
        hive -e "$cart3"
        hive -e "$login4"
        hive -e "$good_click5"
        hive -e "$search_click6"
        hive -e "$orders_goods7"
        hive -e "$orders8"
        hive -e "alter table dwd.dwd_order_orders_incr drop partition (dt = '99990000')"
        clear_data
    ;;
esac
