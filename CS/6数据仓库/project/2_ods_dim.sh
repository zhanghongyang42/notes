#!/bin/bash

# 如果是输入的日期按照取输入日期；如果没输入日期取当前时间的前一天
#获取输入日期2022-02-02 或者 当天日期的前一天日期
if [ -n "$2" ] ;then
    do_dateg=$2
    do_date=${do_dateg//-/}
else
    do_dateg=`date -d '-1 day' +%F`
    do_date=${do_dateg//-/}
fi


station2="
with
    station as (SELECT id,name,sub_company from ods.ods_sys_tbl_sys_station_full WHERE dt=$do_date),
    station_config as (SELECT station_id,collect_set(city_code) citys,collect_set(state_code) states from ods.ods_sys_tbl_sys_station_city_full WHERE dt=$do_date group by station_id)
insert overwrite table dim.dim_common_station_full partition(dt=$do_date)
select
    station.id,
    station.name,
    station.sub_company ,
    station_config.citys ,
    station_config.states
from station
left join station_config  on station.id=station_config.station_id;
"

member3="
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
with
    dim_user_member_zipper_9999 as (select * from dim.dim_user_member_zipper where dt = '99999999'),
    ods_member_tmp as (SELECT * FROM ods.ods_sys_tbl_member_upsert WHERE dt = $do_date),
    ods_profile as (select * from ods.ods_sys_tbl_member_profile_full WHERE deleted = FALSE and dt = $do_date),
    ods_institutes as (SELECT * from ods.ods_sys_tbl_member_institutes_full WHERE dt = $do_date),
    ods_member as (select ods_member_tmp.id       member_id,
                           ods_member_tmp.add_time member_add_time,
                           ods_member_tmp."'`type`'"   member_type,
                           privilege               member_privilege,
                           check_type              member_check_type,
                           ods_profile.name        member_name,
                           sex                     member_sex,
                           birthday                member_birthday,
                           occupational            member_occupa,
                           education               member_educat,
                           income                  member_income,
                           ods_institutes.code     institutes_code,
                           ods_institutes.name     institutes_name,
                           simple_name             institutes_simple_name,
                           ods_institutes."'`type`'"   institutes_type,
                           ods_institutes."'`source`'" institutes_source,
                           task_name               task_group_name,
                           sys_station_id          station_id,
                           county_code             area_id,
                           created                 create_time,
                           modified                operate_time
                    from ods_member_tmp
                    left join ods_profile
                    on ods_member_tmp.id = ods_profile.member_id
                    left join ods_institutes
                    on ods_member_tmp.institutes_id = ods_institutes.id),
    task_name_mid as (SELECT *, row_number() over (partition by id order by modified desc) RK from ods.ods_sys_tbl_member_tasks_upsert),
    ods_task_name_mid as (select member_id, id, name from task_name_mid where member_id is not NULL and rk = 1),
    task_mid as (SELECT *, row_number() over (partition by id order by modified desc) RK from ods.ods_sys_tbl_member_tasks_fundslog_upsert),
    task_mid1 as (SELECT *, row_number() over (partition by member_id,tasks_id order by modified desc) RK1 from task_mid where RK = 1),
    ods_task_mid as (SELECT member_id, tasks_id, remain FROM task_mid1 WHERE rk1 = 1),
    ods_task_table as (SELECT t.member_id member_id,
                               collect_set(named_struct('tasks_id', tasks_id, 'task_name', name, 'remain',remain)) task_attrs
                        from ods_task_mid t
                        left join ods_task_name_mid
                        on t.member_id = ods_task_name_mid.member_id and t.tasks_id = ods_task_name_mid.id
                        WHERE t.member_id is NOT null
                        GROUP BY t.member_id),
    ods_new as (SELECT ods_member.member_id member_id,
                        member_add_time,
                        member_type,
                        member_privilege,
                        member_check_type,
                        member_name,
                        member_sex,
                        member_birthday,
                        member_occupa,
                        member_educat,
                        member_income,
                        institutes_code,
                        institutes_name,
                        institutes_simple_name,
                        institutes_type,
                        institutes_source,
                        task_group_name,
                        task_attrs,
                        station_id,
                        area_id,
                        create_time,
                        operate_time,
                        $do_date           start_date,
                        '99999999'           end_date
                 from ods_member
                 left join ods_task_table
                 on ods_member.member_id = ods_task_table.member_id),
    tmp as (select olda.member_id              old_member_id,
                    olda.member_add_time        old_member_add_time,
                    olda.member_type            old_member_type,
                    olda.member_privilege       old_member_privilege,
                    olda.member_check_type      old_member_check_type,
                    olda.member_name            old_member_name,
                    olda.member_sex             old_member_sex,
                    olda.member_birthday        old_member_birthday,
                    olda.member_occupa          old_member_occupa,
                    olda.member_educat          old_member_educat,
                    olda.member_income          old_member_income,
                    olda.institutes_code        old_institutes_code,
                    olda.institutes_name        old_institutes_name,
                    olda.institutes_simple_name old_institutes_simple_name,
                    olda.institutes_type        old_institutes_type,
                    olda.institutes_source      old_institutes_source,
                    olda.task_group_name        old_task_group_name,
                    olda.task_attrs             old_task_attrs,
                    olda.station_id             old_station_id,
                    olda.area_id                old_area_id,
                    olda.create_time            old_create_time,
                    olda.operate_time           old_operate_time,
                    olda.start_date             old_start_date,
                    olda.end_date               old_end_date,
                    newa.member_id              new_member_id,
                    newa.member_add_time        new_member_add_time,
                    newa.member_type            new_member_type,
                    newa.member_privilege       new_member_privilege,
                    newa.member_check_type      new_member_check_type,
                    newa.member_name            new_member_name,
                    newa.member_sex             new_member_sex,
                    newa.member_birthday        new_member_birthday,
                    newa.member_occupa          new_member_occupa,
                    newa.member_educat          new_member_educat,
                    newa.member_income          new_member_income,
                    newa.institutes_code        new_institutes_code,
                    newa.institutes_name        new_institutes_name,
                    newa.institutes_simple_name new_institutes_simple_name,
                    newa.institutes_type        new_institutes_type,
                    newa.institutes_source      new_institutes_source,
                    newa.task_group_name        new_task_group_name,
                    newa.task_attrs             new_task_attrs,
                    newa.station_id             new_station_id,
                    newa.area_id                new_area_id,
                    newa.create_time            new_create_time,
                    newa.operate_time           new_operate_time,
                    newa.start_date             new_start_date,
                    newa.end_date               new_end_date
            from dim_user_member_zipper_9999 olda
            full outer join ods_new newa
            on olda.member_id = newa.member_id)
insert overwrite table dim.dim_user_member_zipper partition(dt)
select nvl(new_member_id, old_member_id),
       nvl(new_member_add_time, old_member_add_time),
       nvl(new_member_type, old_member_type),
       nvl(new_member_privilege, old_member_privilege),
       nvl(new_member_check_type, old_member_check_type),
       nvl(new_member_name, old_member_name),
       nvl(new_member_sex, old_member_sex),
       nvl(new_member_birthday, old_member_birthday),
       nvl(new_member_occupa, old_member_occupa),
       nvl(new_member_educat, old_member_educat),
       nvl(new_member_income, old_member_income),
       nvl(new_institutes_code, old_institutes_code),
       nvl(new_institutes_name, old_institutes_name),
       nvl(new_institutes_simple_name, old_institutes_simple_name),
       nvl(new_institutes_type, old_institutes_type),
       nvl(new_institutes_source, old_institutes_source),
       nvl(new_task_group_name, old_task_group_name),
       nvl(new_task_attrs, old_task_attrs),
       nvl(new_station_id, old_station_id),
       nvl(new_area_id, old_area_id),
       nvl(new_create_time, old_create_time),
       nvl(new_operate_time, old_operate_time),
       nvl(new_start_date, old_start_date),
       nvl(new_end_date, old_end_date),
       nvl(new_end_date, old_end_date) dt
from tmp
union all
select old_member_id,
       old_member_add_time,
       old_member_type,
       old_member_privilege,
       old_member_check_type,
       old_member_name,
       old_member_sex,
       old_member_birthday,
       old_member_occupa,
       old_member_educat,
       old_member_income,
       old_institutes_code,
       old_institutes_name,
       old_institutes_simple_name,
       old_institutes_type,
       old_institutes_source,
       old_task_group_name,
       old_task_attrs,
       old_station_id,
       old_area_id,
       old_create_time,
       old_operate_time,
       old_start_date,
       regexp_replace(cast(date_add('$do_dateg', -1) as string), '-', ''),
       regexp_replace(cast(date_add('$do_dateg', -1) as string), '-', '') dt
from tmp
where new_member_id is not null and old_member_id is not null;
"

supplier4="
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
with
    dim_user_supplier_zipper_9999 as (select * from dim.dim_user_supplier_zipper where dt = '99999999'),
    ods_supplier_tmp as (SELECT id,
                                 "'`type`'",
                                 add_time,
                                 margin_id,
                                 company_name,
                                 county,
                                 shop_type,
                                 mode,
                                 security_account,
                                 cert_type,
                                 station_id,
                                 hot_sale_amount,
                                 week_sale_amount,
                                 month_sale_amount,
                                 product_hits,
                                 product_score,
                                 service_score,
                                 delivery_score,
                                 org_code,
                                 property,
                                 not_local,
                                 grade_id,
                                 total_growth,
                                 modified,
                                 created
                          FROM ods.ods_sys_tbl_supplier_upsert
                          WHERE dt = $do_date),
    supplier_shop_tmp as (SELECT *, row_number() over (partition by id order by modified desc) RK from ods.ods_sys_tbl_supplier_shop_upsert),
    ods_supplier_shop_tmp as (SELECT "'`type`'", name, supplier_id, banner_title, seo_keyword, add_time, hits FROM supplier_shop_tmp WHERE rk = 1),
    supplier_danger_tmp as (SELECT *, row_number() over (partition by id order by modified desc) RK from ods.ods_sys_tbl_supplier_danger_cert_upsert),
    ods_supplier_danger_tmp as (SELECT supplier_id, collect_set(code) codes from supplier_danger_tmp WHERE RK = 1 GROUP by supplier_id),
    ods_new as (SELECT ods_supplier_tmp.id            supplier_id_new,
                        ods_supplier_tmp.add_time      supplier_add_time_new,
                        ods_supplier_tmp.company_name  supplier_company_name_new,
                        margin_id                      supplier_margin_new,
                        ods_supplier_tmp."'`type`'"        supplier_type_new,
                        mode                           supplier_mode_new,
                        cert_type                      supplier_cert_type_new,
                        security_account               supplier_security_account_new,
                        grade_id                       supplier_grade_id_new,
                        total_growth                   supplier_total_growth_new,
                        product_score                  supplier_product_score_new,
                        service_score                  supplier_service_score_new,
                        delivery_score                 supplier_delivery_score_new,
                        hot_sale_amount                supplier_hot_sale_amount_new,
                        ods_supplier_shop_tmp."'`type`'"   shop_type_new,
                        ods_supplier_shop_tmp.name     shop_name_new,
                        ods_supplier_shop_tmp.add_time shop_add_time_new,
                        org_code                       supplier_org_code_new,
                        property                       supplier_property_new,
                        not_local                      supplier_not_local_new,
                        banner_title                   shop_banner_title_new,
                        seo_keyword                    shop_seo_keyword_new,
                        week_sale_amount               supplier_week_sale_amount_new,
                        month_sale_amount              supplier_month_sale_amount_new,
                        product_hits                   supplier_product_hits_new,
                        hits                           shop_hits_new,
                        codes                          supplier_codes_new,
                        county                         supplier_area_id_new,
                        station_id                     supplier_station_id_new,
                        created                        create_time_new,
                        modified                       operate_time_new,
                        $do_date                     start_date_new,
                        '99999999'                     end_date_new
                 FROM ods_supplier_tmp
                 left join ods_supplier_shop_tmp
                 on ods_supplier_tmp.id = ods_supplier_shop_tmp.supplier_id
                 left join ods_supplier_danger_tmp
                 on ods_supplier_tmp.id = ods_supplier_danger_tmp.supplier_id),
    tmp as (select * from dim_user_supplier_zipper_9999 olda full outer join ods_new newa on olda.supplier_id = newa.supplier_id_new)
insert overwrite table dim.dim_user_supplier_zipper partition(dt)
select nvl(supplier_id_new, supplier_id),
       nvl(supplier_add_time_new, supplier_add_time),
       nvl(supplier_company_name_new, supplier_company_name),
       nvl(supplier_margin_new, supplier_margin),
       nvl(supplier_type_new, supplier_type),
       nvl(supplier_mode_new, supplier_mode),
       nvl(supplier_cert_type_new, supplier_cert_type),
       nvl(supplier_security_account_new, supplier_security_account),
       nvl(supplier_grade_id_new, supplier_grade_id),
       nvl(supplier_total_growth_new, supplier_total_growth),
       nvl(supplier_product_score_new, supplier_product_score),
       nvl(supplier_service_score_new, supplier_service_score),
       nvl(supplier_delivery_score_new, supplier_delivery_score),
       nvl(supplier_hot_sale_amount_new, supplier_hot_sale_amount),
       nvl(shop_type_new, shop_type),
       nvl(shop_name_new, shop_name),
       nvl(shop_add_time_new, shop_add_time),
       nvl(supplier_org_code_new, supplier_org_code),
       nvl(supplier_property_new, supplier_property),
       nvl(supplier_not_local_new, supplier_not_local),
       nvl(shop_banner_title_new, shop_banner_title),
       nvl(shop_seo_keyword_new, shop_seo_keyword),
       nvl(supplier_week_sale_amount_new, supplier_week_sale_amount),
       nvl(supplier_month_sale_amount_new, supplier_month_sale_amount),
       nvl(supplier_product_hits_new, supplier_product_hits),
       nvl(shop_hits_new, shop_hits),
       nvl(supplier_codes_new, supplier_codes),
       nvl(supplier_area_id_new, supplier_area_id),
       nvl(supplier_station_id_new, supplier_station_id),
       nvl(create_time_new, create_time),
       nvl(operate_time_new, operate_time),
       nvl(start_date_new, start_date),
       nvl(end_date_new, end_date),
       nvl(end_date_new, end_date) dt
from tmp
union all
select supplier_id,
       supplier_add_time,
       supplier_company_name,
       supplier_margin,
       supplier_type,
       supplier_mode,
       supplier_cert_type,
       supplier_security_account,
       supplier_grade_id,
       supplier_total_growth,
       supplier_product_score,
       supplier_service_score,
       supplier_delivery_score,
       supplier_hot_sale_amount,
       shop_type,
       shop_name,
       shop_add_time,
       supplier_org_code,
       supplier_property,
       supplier_not_local,
       shop_banner_title,
       shop_seo_keyword,
       supplier_week_sale_amount,
       supplier_month_sale_amount,
       supplier_product_hits,
       shop_hits,
       supplier_codes,
       supplier_area_id,
       supplier_station_id,
       create_time,
       operate_time,
       start_date,
       regexp_replace(cast(date_add('$do_dateg', -1) as string), '-', ''),
       regexp_replace(cast(date_add('$do_dateg', -1) as string), '-', '') dt
from tmp
where supplier_id is not null and supplier_id_new is not null;
"

product5="
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
set mapred.map.tasks=30;
set mapred.reduce.tasks=20;
with
dim_product_product_zipper_9999 as (select * from dim.dim_product_product_zipper where dt='99999999'),

ods_product_tmp as (SELECT id,code,cate_id,brand_id,type_id,supplier_id,name,mkt_price,price,temp_price,img,stock_amount,sale_amount,sale_time,alert_amount,usable_amount,no_stock,maker,
					sales_by_proxy,delivery_cycle,station_id,hot_sale_status,cas_code,bissness_type,created,modified,instructions_status,price_update_time
				FROM ods.ods_product_tbl_product_basic_upsert WHERE dt='$do_date'),

cate_tmp0 as (SELECT id,split(CONCAT(pids,string(id)),':')[6] id5,split(CONCAT(pids,string(id)),':')[5] id4,split(CONCAT(pids,string(id)),':')[4] id3,split(CONCAT(pids,string(id)),':')[3] id2,split(CONCAT(pids,string(id)),':')[2] id1
				FROM ods.ods_product_tbl_category_full WHERE dt='$do_date'),
cate_tmp1 as (SELECT id id1,name name1,tax_name tax_name1 FROM ods.ods_product_tbl_category_full WHERE dt='$do_date'),
cate_tmp2 as (SELECT id id2,name name2,tax_name tax_name2 FROM ods.ods_product_tbl_category_full WHERE dt='$do_date'),
cate_tmp3 as (SELECT id id3,name name3,tax_name tax_name3 FROM ods.ods_product_tbl_category_full WHERE dt='$do_date'),
cate_tmp4 as (SELECT id id4,name name4,tax_name tax_name4 FROM ods.ods_product_tbl_category_full WHERE dt='$do_date'),
cate_tmp5 as (SELECT id id5,name name5,tax_name tax_name5 FROM ods.ods_product_tbl_category_full WHERE dt='$do_date'),
ods_cate_tmp as (SELECT id,cate_tmp1.id1,name1,tax_name1,cate_tmp2.id2,name2,tax_name2,cate_tmp3.id3,name3,tax_name3,cate_tmp4.id4,name4,tax_name4,cate_tmp5.id5,name5,tax_name5
				FROM cate_tmp0 left join cate_tmp1 on cate_tmp0.id1=cate_tmp1.id1 LEFT join cate_tmp2 on cate_tmp0.id2=cate_tmp2.id2 LEFT join cate_tmp3 on cate_tmp0.id3=cate_tmp3.id3
				LEFT join cate_tmp4 on cate_tmp0.id4=cate_tmp4.id4 LEFT join cate_tmp5 on cate_tmp0.id5=cate_tmp5.id5),

ods_type_tmp as (SELECT id,name from ods.ods_product_tbl_product_type_full WHERE dt='$do_date' ),
ods_dangerous_tmp as (SELECT name,cas_code,sub_name,cate_name,description,sale_count from ods.ods_product_tbl_product_dangerous_full WHERE dt='$do_date'),

brand_tmp as (SELECT *,row_number() over(partition by id order by modified desc) RK from ods.ods_product_tbl_brand_upsert),
ods_brand_tmp as (SELECT id,name,recommend,reseller_level,main_id,mained FROM brand_tmp WHERE rk=1),

ods_agent_tmp0 as (SELECT *,row_number() over(partition by id order by modified desc) RK from ods.ods_sys_tbl_supplier_brand_agent_upsert),
ods_agent_tmp1 as (SELECT supplier_id,brand_id,level,area ,modified from ods_agent_tmp0 WHERE rk=1),
ods_agent_tmp as (SELECT supplier_id,brand_id,level,area
				from (
						SELECT supplier_id,brand_id,level,area ,row_number() over(PARTITION by supplier_id,brand_id ORDER by modified) rk
						from ods_agent_tmp1
						) agent_tmp0
				WHERE agent_tmp0.rk=1),


tag_relt_tmp0 as (SELECT *,row_number() over(partition by id order by modified desc) RK from ods.ods_product_tbl_product_relate_tag_upsert),
tag_relt_tmp as (SELECT product_id,tag_id from tag_relt_tmp0 WHERE rk=1),
tag_tmp0 as (SELECT id,name,weight,row_number() over(partition by id order by modified desc) RK from ods.ods_product_tbl_product_tag_upsert),
tag_tmp as (select id,name,weight from tag_tmp0 WHERE rk=1),
ods_prod_tag_tmp0 as (SELECT product_id,tag_id,name,weight FROM tag_relt_tmp left JOIN tag_tmp on tag_relt_tmp.tag_id=tag_tmp.id),
ods_prod_tag_tmp as (SELECT product_id,collect_set(named_struct('tag_id',tag_id,'name',name,'weight',weight)) tag_attrs
						FROM ods_prod_tag_tmp0
						group by product_id),

ods_new as (SELECT
				ods_product_tmp.id product_id_new,
				CONCAT(CONCAT(ods_product_tmp.brand_id,'_'),ods_product_tmp.code) product_spu_new,
				ods_product_tmp.name product_name_new,
				ods_product_tmp.code product_code_new,
				ods_product_tmp.cas_code product_cas_new,
				ods_product_tmp.supplier_id product_supplier_id_new,
				mkt_price product_mkt_price_new,
				ods_product_tmp.price product_price_new,
				alert_amount product_alert_amount_new,
				stock_amount product_stock_amount_new,
				usable_amount product_usable_amount_new,
				ods_product_tmp.maker product_maker_new,
				bissness_type product_bissness_type_new,
				sales_by_proxy product_sales_by_proxy_new,
				delivery_cycle product_delivery_cycle_new,
				CAST(no_stock AS string) product_no_stock_new,
				hot_sale_status product_hot_sale_status_new,
				ods_product_tmp.type_id product_type_id_new,
				ods_product_tmp.brand_id product_brand_id_new,
				ods_product_tmp.cate_id product_cate_id_new,
				ods_product_tmp.img product_img_new,
				price_update_time product_price_update_time_new,
				ods_product_tmp.sale_time product_sale_time_new,
				CAST(instructions_status AS string) product_instructions_new,
				ods_product_tmp.station_id product_station_id_new,
				ods_brand_tmp.name brand_name_new,
				ods_brand_tmp.recommend brand_recommend_new,
				ods_brand_tmp.reseller_level brand_level_new,
				id1 cate_one_id_new,
				name1 cate_one_name_new,
				tax_name1 cate_one_tax_name_new,
				id2 cate_two_id_new,
				name2 cate_two_name_new,
				tax_name2 cate_two_tax_name_new,
				id3 cate_three_id_new,
				name3 cate_three_name_new,
				tax_name3 cate_three_tax_name_new,
				id4 cate_four_id_new,
				name4 cate_four_name_new,
				tax_name4 cate_four_tax_name_new,
				id5 cate_five_id_new,
				name5 cate_five_name_new,
				tax_name5 cate_five_tax_name_new,
				ods_type_tmp.name type_name_new,
				tag_attrs tag_attrs_new,
				ods_dangerous_tmp.name dangerous_name_new,
				ods_dangerous_tmp.sub_name dangerous_sub_name_new,
				ods_dangerous_tmp.cate_name dangerous_cate_name_new,
				ods_dangerous_tmp.description dangerous_description_new,
				ods_agent_tmp.level supplier_brand_agent_level_new,
				ods_agent_tmp.area supplier_brand_agent_area_new,
				ods_product_tmp.created create_time_new,
				ods_product_tmp.modified operate_time_new,
    		'$do_date' start_date_new,
    		'99999999' end_date_new
			from  ods_product_tmp
			left join ods_brand_tmp
			on ods_product_tmp.brand_id=ods_brand_tmp.id
			LEFT JOIN ods_cate_tmp
			on ods_product_tmp.cate_id=ods_cate_tmp.id
			LEFT JOIN ods_type_tmp
			on ods_product_tmp.type_id=ods_type_tmp.id
			left join ods_prod_tag_tmp
			on ods_product_tmp.id=ods_prod_tag_tmp.product_id
			left JOIN ods_dangerous_tmp
			on ods_product_tmp.cas_code=ods_dangerous_tmp.cas_code
			LEFT JOIN ods_agent_tmp
			on ods_product_tmp.supplier_id=ods_agent_tmp.supplier_id and ods_product_tmp.brand_id=ods_agent_tmp.brand_id),

tmp as (select * from dim_product_product_zipper_9999 olda full outer join ods_new newa on olda.product_id=newa.product_id_new)


insert overwrite table dim.dim_product_product_zipper partition(dt)
select
	nvl(product_id_new,product_id),
	nvl(product_spu_new,product_spu),
	nvl(product_name_new,product_name),
	nvl(product_code_new,product_code),
	nvl(product_cas_new,product_cas),
	nvl(product_supplier_id_new,product_supplier_id),
	nvl(product_mkt_price_new,product_mkt_price),
	nvl(product_price_new,product_price),
	nvl(product_alert_amount_new,product_alert_amount),
	nvl(product_stock_amount_new,product_stock_amount),
	nvl(product_usable_amount_new,product_usable_amount),
	nvl(product_maker_new,product_maker),
	nvl(product_bissness_type_new,product_bissness_type),
	nvl(product_sales_by_proxy_new,product_sales_by_proxy),
	nvl(product_delivery_cycle_new,product_delivery_cycle),
	nvl(product_no_stock_new,product_no_stock),
	nvl(product_hot_sale_status_new,product_hot_sale_status),
	nvl(product_type_id_new,product_type_id),
	nvl(product_brand_id_new,product_brand_id),
	nvl(product_cate_id_new,product_cate_id),
	nvl(product_img_new,product_img),
	nvl(product_price_update_time_new,product_price_update_time),
	nvl(product_sale_time_new,product_sale_time),
	nvl(product_instructions_new,product_instructions),
	nvl(product_station_id_new,product_station_id),
	nvl(brand_name_new,brand_name),
	nvl(brand_recommend_new,brand_recommend),
	nvl(brand_level_new,brand_level),
	nvl(cate_one_id_new,cate_one_id),
	nvl(cate_one_name_new,cate_one_name),
	nvl(cate_one_tax_name_new,cate_one_tax_name),
	nvl(cate_two_id_new,cate_two_id),
	nvl(cate_two_name_new,cate_two_name),
	nvl(cate_two_tax_name_new,cate_two_tax_name),
	nvl(cate_three_id_new,cate_three_id),
	nvl(cate_three_name_new,cate_three_name),
	nvl(cate_three_tax_name_new,cate_three_tax_name),
	nvl(cate_four_id_new,cate_four_id),
	nvl(cate_four_name_new,cate_four_name),
	nvl(cate_four_tax_name_new,cate_four_tax_name),
	nvl(cate_five_id_new,cate_five_id),
	nvl(cate_five_name_new,cate_five_name),
	nvl(cate_five_tax_name_new,cate_five_tax_name),
	nvl(type_name_new,type_name),
	nvl(tag_attrs_new,tag_attrs),
	nvl(dangerous_name_new,dangerous_name),
	nvl(dangerous_sub_name_new,dangerous_sub_name),
	nvl(dangerous_cate_name_new,dangerous_cate_name),
	nvl(dangerous_description_new,dangerous_description),
	nvl(supplier_brand_agent_level_new,supplier_brand_agent_level),
	nvl(supplier_brand_agent_area_new,supplier_brand_agent_area),
	nvl(create_time_new,create_time),
	nvl(operate_time_new,operate_time),
	nvl(start_date_new,start_date),
	nvl(end_date_new,end_date),
	nvl(end_date_new,end_date) dt
from tmp
union all
select
	product_id,
	product_spu,
	product_name,
	product_code,
	product_cas,
	product_supplier_id,
	product_mkt_price,
	product_price,
	product_alert_amount,
	product_stock_amount,
	product_usable_amount,
	product_maker,
	product_bissness_type,
	product_sales_by_proxy,
	product_delivery_cycle,
	product_no_stock,
	product_hot_sale_status,
	product_type_id,
	product_brand_id,
	product_cate_id,
	product_img,
	product_price_update_time,
	product_sale_time,
	product_instructions,
	product_station_id,
	brand_name,
	brand_recommend,
	brand_level,
	cate_one_id,
	cate_one_name,
	cate_one_tax_name,
	cate_two_id,
	cate_two_name,
	cate_two_tax_name,
	cate_three_id,
	cate_three_name,
	cate_three_tax_name,
	cate_four_id,
	cate_four_name,
	cate_four_tax_name,
	cate_five_id,
	cate_five_name,
	cate_five_tax_name,
	type_name,
	tag_attrs,
	dangerous_name,
	dangerous_sub_name,
	dangerous_cate_name,
	dangerous_description,
	supplier_brand_agent_level,
	supplier_brand_agent_area,
	create_time,
	operate_time,
	start_date,
	regexp_replace(cast(date_add('$do_dateg',-1) as string),'-',''),
	regexp_replace(cast(date_add('$do_dateg',-1) as string),'-','') dt
from tmp
where product_id is not null and product_id_new is not null;
"

case $1 in
    product){
        hive -e "$product5"
        }
    ;;
    member){
        hive -e "$member3"
        }
    ;;
    supplier){
        hive -e "$supplier4"
        }
    ;;
    all){
        hive -e "$station2"
        hive -e "$member3"
        hive -e "$supplier4"
        hive -e "$product5"
        }
    ;;
esac






















