#!/bin/bash

#20220309 20220220
date_1=`date -d '-1 day' +%F`
date1=${date_1//-/}
#20220303 20220213
date_7=`date -d '-7 day' +%F`
date7=${date_7//-/}
#20220208 20220120
date_30=`date -d '-30 day' +%F`
date30=${date_30//-/}
#20210310 20210220
date_365=`date -d '-365 day' +%F`
date365=${date_365//-/}

member="
with
    member1_tmp as(SELECT
                       member_id,
                       cart_count cart_1d_count,
                       click_count click_1d_count,
                       login_count login_1d_count,
                       favorites_count favorites_1d_count,
                       search_count search_1d_count,
                       orders_count orders_1d_count,
                       orders_price orders_1d_price,
                       orders_all_price orders_1d_all_price
                   FROM dwi.dwi_user_member_day WHERE dt='$date1'),
    member7_tmp as (SELECT
                        member_id,
                        SUM(cart_count) cart_7d_count,
                        SUM(click_count) click_7d_count,
                        SUM(login_count) login_7d_days,
                        SUM(favorites_count) favorites_7d_count,
                        SUM(search_count) search_7d_count,
                        SUM(orders_count) orders_7d_count,
                        SUM(orders_price) orders_7d_price,
                        SUM(orders_all_price) orders_7d_all_price
                    FROM dwi.dwi_user_member_day WHERE dt<='$date1' and dt>='$date7'
                    group by member_id),
    member30_tmp as (select
                         member_id,
                         SUM(click_count) click_30d_count,
                         SUM(login_count) login_30d_count,
                         SUM(search_count) search_30d_count,
                         SUM(orders_price) orders_30d_price,
                         SUM(orders_count) orders_30d_count,
                         SUM(orders_all_price) orders_30d_all_price,
                         sum(if(login_count>0,1,0)) login_30d_days,
                         sum(if(orders_count>0,1,0)) orders_30d_days
                     FROM dwi.dwi_user_member_day
                     WHERE dt<='$date1' and dt>='$date30'
                     group by member_id),
    member_long_tmp as (SELECT
                            member_id,
                            SUM(click_count) click_total_count,
                            sum(if(login_count>0,1,0)) login_total_days,
                            max(if(orders_count>0,dt,0)) orders_last_date,
                            max(if(login_count>0,dt,0)) login_last_date
                        from dwi.dwi_user_member_day
                        WHERE dt<='$date1' and dt>='$date365'
                        group by member_id)
insert overwrite table dws.dws_user_member_wide partition(dt='$date1')
select
    coalesce(member1_tmp.member_id,member7_tmp.member_id,member30_tmp.member_id,member_long_tmp.member_id),
    cart_1d_count,
    cart_7d_count,
    click_1d_count,
    click_7d_count,
    click_30d_count,
    click_total_count,
    login_1d_count,
    login_30d_count,
    login_30d_days,
    login_7d_days,
    login_total_days,
    login_last_date,
    favorites_1d_count,
    favorites_7d_count,
    search_1d_count,
    search_7d_count,
    search_30d_count,
    orders_1d_count,
    orders_7d_count,
    orders_30d_count,
    orders_30d_days,
    orders_last_date,
    orders_1d_price,
    orders_7d_price,
    orders_30d_price,
    orders_1d_all_price,
    orders_7d_all_price,
    orders_30d_all_price
FROM member1_tmp
full outer join member7_tmp
on member1_tmp.member_id=member7_tmp.member_id
full outer join member30_tmp
on nvl(member1_tmp.member_id,member7_tmp.member_id)=member30_tmp.member_id
full outer join member_long_tmp
on coalesce(member1_tmp.member_id,member7_tmp.member_id,member30_tmp.member_id)=member_long_tmp.member_id
;
"
hive -e "$member"


supplier="
with
    member1_tmp as(SELECT
                       supplier_id,
                       cart_count cart_1d_count,
                       click_count click_1d_count,
                       favorites_count favorites_1d_count,
                       click_search_count search_1d_count,
                       orders_count orders_1d_count,
                       orders_price orders_1d_price,
                       orders_all_price orders_1d_all_price
                   FROM dwi.dwi_user_supplier_day WHERE dt='$date1'),
    member7_tmp as (SELECT
                        supplier_id,
                        SUM(cart_count) cart_7d_count,
                        SUM(click_count) click_7d_count,
                        SUM(favorites_count) favorites_7d_count,
                        SUM(click_search_count) search_7d_count,
                        SUM(orders_count) orders_7d_count,
                        SUM(orders_price) orders_7d_price,
                        SUM(orders_all_price) orders_7d_all_price
                    FROM dwi.dwi_user_supplier_day WHERE dt<='$date1' and dt>='$date7'
                    group by supplier_id),
    member30_tmp as (select
                         supplier_id,
                         SUM(click_count) click_30d_count,
                         SUM(click_search_count) search_30d_count,
                         SUM(orders_price) orders_30d_price,
                         SUM(orders_count) orders_30d_count,
                         SUM(orders_all_price) orders_30d_all_price,
                         sum(if(orders_count>0,1,0)) orders_30d_days
                     FROM dwi.dwi_user_supplier_day
                     WHERE dt<='$date1' and dt>='$date30'
                     group by supplier_id),
    member_long_tmp as (SELECT
                            supplier_id,
                            SUM(click_count) click_total_count,
                            max(if(orders_count>0,dt,0)) orders_last_date
                        from dwi.dwi_user_supplier_day
                        WHERE dt<='$date1' and dt>='$date365'
                        group by supplier_id)
insert overwrite table dws.dws_user_supplier_wide partition(dt='$date1')
select
    coalesce(member1_tmp.supplier_id,member7_tmp.supplier_id,member30_tmp.supplier_id,member_long_tmp.supplier_id),
    cart_1d_count,
    cart_7d_count,
    click_1d_count,
    click_7d_count,
    click_30d_count,
    click_total_count,
    favorites_1d_count,
    favorites_7d_count,
    search_1d_count,
    search_7d_count,
    search_30d_count,
    orders_1d_count,
    orders_7d_count,
    orders_30d_count,
    orders_30d_days,
    orders_last_date,
    orders_1d_price,
    orders_7d_price,
    orders_30d_price,
    orders_1d_all_price,
    orders_7d_all_price,
    orders_30d_all_price
FROM member1_tmp
full outer join member7_tmp
on member1_tmp.supplier_id=member7_tmp.supplier_id
full outer join member30_tmp
on nvl(member1_tmp.supplier_id,member7_tmp.supplier_id) =member30_tmp.supplier_id
full outer join member_long_tmp
on coalesce(member1_tmp.supplier_id,member7_tmp.supplier_id,member30_tmp.supplier_id)=member_long_tmp.supplier_id
;
"
hive -e "$supplier"

product="
with
    member1_tmp as(SELECT
                       product_id,
                       product_spu,
                       cart_count cart_1d_count,
                       cart_amount cart_1d_amount,
                       click_count click_1d_count,
                       favorites_count favorites_1d_count,
                       click_search_count search_1d_count,
                       orders_goods_count orders_1d_count,
                       orders_goods_amount orders_1d_amount
                   FROM dwi.dwi_product_product_day WHERE dt='$date1'),
    member7_tmp as (SELECT
                        product_id,
                        max(product_spu) product_spu,
                        SUM(cart_count) cart_7d_count,
                        SUM(cart_amount) cart_7d_amount,
                        SUM(click_count) click_7d_count,
                        SUM(favorites_count) favorites_7d_count,
                        SUM(click_search_count) search_7d_count,
                        SUM(orders_goods_count) orders_7d_count,
                        sum(orders_goods_amount) orders_7d_amount
                    FROM dwi.dwi_product_product_day WHERE dt<='$date1' and dt>='$date7'
                    group by product_id),
    member30_tmp as (select
                         product_id,
                         max(product_spu) product_spu,
                         SUM(click_count) click_30d_count,
                         SUM(click_search_count) search_30d_count,
                         SUM(orders_goods_count) orders_30d_count,
                         SUM(orders_goods_amount) orders_30d_amount,
                         sum(if(orders_goods_count>0,1,0)) orders_30d_days
                     FROM dwi.dwi_product_product_day
                     WHERE dt<='$date1' and dt>='$date30'
                     group by product_id),
    member_long_tmp as (SELECT
                            product_id,
                            max(product_spu) product_spu,
                            SUM(click_count) click_total_count,
                            max(if(orders_goods_count>0,dt,0)) orders_last_date
                        from dwi.dwi_product_product_day
                        WHERE dt<='$date1' and dt>='$date365'
                        group by product_id)
insert overwrite table dws.dws_product_product_wide partition(dt='$date1')
select
    coalesce(member1_tmp.product_id,member7_tmp.product_id,member30_tmp.product_id,member_long_tmp.product_id),
    coalesce(member1_tmp.product_spu,member7_tmp.product_spu,member30_tmp.product_spu,member_long_tmp.product_spu),
    cart_1d_count,
    cart_7d_count,
    cart_1d_amount,
    cart_7d_amount,
    favorites_1d_count,
    favorites_7d_count,
    click_1d_count,
    click_7d_count,
    click_30d_count,
    click_total_count,
    search_1d_count,
    search_7d_count,
    search_30d_count,
    orders_1d_count,
    orders_7d_count,
    orders_30d_count,
    orders_30d_days,
    orders_last_date,
    orders_1d_amount,
    orders_7d_amount,
    orders_30d_amount
FROM member1_tmp
full outer join member7_tmp
on member1_tmp.product_id=member7_tmp.product_id
full outer join member30_tmp
on nvl(member1_tmp.product_id,member7_tmp.product_id)=member30_tmp.product_id
full outer join member_long_tmp
on coalesce(member1_tmp.product_id,member7_tmp.product_id,member30_tmp.product_id)=member_long_tmp.product_id
;
"

hive -e "$product"
















