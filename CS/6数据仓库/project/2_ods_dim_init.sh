#!/bin/bash

# 时间维表比较特殊，是python生成的时间数据，直接load进时间维表，只需手动加载一次

date_ods_first='20220218'

area_create1="
DROP TABLE IF EXISTS dim.dim_common_area_once;
CREATE TABLE dim.dim_common_area_once(
    "'`area_id`'" STRING COMMENT '区id' ,
    "'`county_code`'" STRING COMMENT '区编码' ,
    "'`county_name`'" STRING COMMENT '区名称' ,
    "'`city_id`'" STRING COMMENT '城市id' ,
    "'`city_code`'" STRING COMMENT '城市编码' ,
    "'`city_name`'" STRING COMMENT '城市名称' ,
    "'`state_id`'" STRING COMMENT '省id' ,
    "'`state_code`'" STRING COMMENT '省编码' ,
    "'`state_name`'" STRING COMMENT '省名称'
    ) comment '地区维表'
STORED AS PARQUET;
"

area1="
with
    county as (select id,code,name,standard_county_code,city_code from ods.ods_sys_tbl_sys_county_once WHERE deleted=FALSE),
    city as (select id,code,name,standard_city_code,state_code from ods.ods_sys_tbl_sys_city_once WHERE deleted=FALSE),
    state as (select id,code,name,standard_state_code,country_code from ods.ods_sys_tbl_sys_state_once WHERE deleted=FALSE)
insert overwrite table dim.dim_common_area_once
select
    county.id,
    county.code,
    county.name,
    city.id,
    city.code,
    city.name,
    state.id ,
    state.code ,
    state.name
from county
left join city on county.city_code = city.code
LEFT JOIN state on city.state_code = state.code ;
"

station_create2="
DROP TABLE IF EXISTS dim.dim_common_station_full;
CREATE TABLE dim.dim_common_station_full(
    "'`station_id`'" STRING COMMENT '配送区id',
    "'`station_name`'" STRING COMMENT '配送区名称',
    "'`station_company`'" string COMMENT '配送区所属公司名称',
    "'`city_codes`'" ARRAY<STRING> COMMENT '该配送区范围内城市编码列表',
    "'`state_codes`'" ARRAY<STRING> COMMENT '该配送区范围内省编码列表'
)
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

station2="
with
    station as (SELECT id,name,sub_company from ods.ods_sys_tbl_sys_station_full WHERE dt=$date_ods_first),
    station_config as (SELECT station_id,collect_set(city_code) citys,collect_set(state_code) states from ods.ods_sys_tbl_sys_station_city_full WHERE dt=$date_ods_first group by station_id)
insert overwrite table dim.dim_common_station_full partition(dt=$date_ods_first)
select
    station.id,
    station.name,
    station.sub_company ,
    station_config.citys ,
    station_config.states
from station
left join station_config
on station.id=station_config.station_id;
"

member_create3="
DROP TABLE IF EXISTS dim.dim_user_member_zipper;
CREATE TABLE dim.dim_user_member_zipper(
    "'`member_id`'" STRING comment '会员id',
    "'`member_add_time`'" STRING comment '注册时间',
    "'`member_type`'" STRING comment '会员类型',
    "'`member_privilege`'" STRING comment '会员权限',
    "'`member_check_type`'" STRING comment '会员验收方式',
    "'`member_name`'" STRING comment '用户名',
    "'`member_sex`'" STRING comment '用户性别',
    "'`member_birthday`'" STRING comment '用户生日',
    "'`member_occupa`'" STRING comment '用户职业',
    "'`member_educat`'" STRING comment '用户学历',
    "'`member_income`'" STRING comment '用户年收入',
    "'`institutes_code`'" STRING comment '研究所编码',
    "'`institutes_name`'" STRING comment '研究所名称',
    "'`institutes_simple_name`'" STRING comment '研究所简称',
    "'`institutes_type`'" STRING comment '研究所类型',
    "'`institutes_source`'" STRING comment '研究所来源',
    "'`task_group_name`'" STRING comment '课题组名称',
    "'`task_attrs`'" ARRAY<STRUCT<tasks_id:bigint,task_name:STRING,remain:double>> COMMENT '课题号属性',
    "'`station_id`'" STRING comment '会员所属配送区ID',
    "'`area_id`'" STRING comment '会员所在地id',
    "'`create_time`'" STRING comment '创建时间',
    "'`operate_time`'" STRING comment '操作时间',
    "'`start_date`'" STRING comment '开始日期',
    "'`end_date`'" STRING comment '结束日期'
)
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

# ods_sys_tbl_member_profile_full 的member_id 是唯一的，与ods_sys_tbl_member_upsert是一对一的关系
member3="
with
    member_tmp as (SELECT * FROM ods.ods_sys_tbl_member_upsert WHERE dt=$date_ods_first),
    profile as (select * from ods.ods_sys_tbl_member_profile_full WHERE dt=$date_ods_first),
    institutes as (SELECT * from ods.ods_sys_tbl_member_institutes_full WHERE dt=$date_ods_first),
    task_name_mid as (select member_id,id,name
                      from ods.ods_sys_tbl_member_tasks_upsert where valid=1 and audit_status=1 and member_id is not NULL  and dt=$date_ods_first),
    task_table as (SELECT t.member_id,collect_set(named_struct('tasks_id',tasks_id,'task_name',name,'remain',remain)) task_attrs
                   from (
                            SELECT member_id,tasks_id,remain, RANK() OVER(PARTITION BY member_id,tasks_id ORDER BY modified,created DESC) RK
                            FROM ods.ods_sys_tbl_member_tasks_fundslog_upsert WHERE dt='20220218' -- 首日之前失效的数据不要了,只要截止到0218的最新剩余金额
                        ) t
                   left join task_name_mid
                   on t.member_id=task_name_mid.member_id and t.tasks_id=task_name_mid.id
                   WHERE t.RK=1 GROUP BY t.member_id )
insert overwrite table dim.dim_user_member_zipper partition(dt='99999999')
select
    member_tmp.id ,
    member_tmp.add_time,
    member_tmp."'`type`'",
    privilege,
    check_type,
    profile.name,
    sex,
    birthday,
    occupational,
    education,
    income,
    institutes.code,
    institutes.name,
    simple_name,
    institutes."'`type`'",
    institutes."'`source`'",
    task_name,
    task_attrs,
    sys_station_id,
    county_code,
    created,
    modified,
    $date_ods_first,
    '99999999'
from member_tmp
left join profile  on member_tmp.id=profile.member_id
left join institutes  on member_tmp.institutes_id=institutes.id
left join task_table  on member_tmp.id=task_table.member_id;
"

supplier_create4="
DROP TABLE IF EXISTS dim.dim_user_supplier_zipper;
CREATE TABLE dim.dim_user_supplier_zipper(
    "'`supplier_id`'" STRING comment '商家id',
    "'`supplier_add_time`'" STRING comment '商家注册时间',
    "'`supplier_company_name`'" STRING comment '商家公司名称',
    "'`supplier_margin`'" STRING comment '保证金标准',
    "'`supplier_type`'" STRING comment '商家类型',
    "'`supplier_mode`'" STRING comment '商家合作模式',
    "'`supplier_cert_type`'" STRING comment '商家资质类型',
    "'`supplier_security_account`'" DECIMAL(16,2) comment '商家保证金',
    "'`supplier_grade_id`'" STRING comment '商家等级',
    "'`supplier_total_growth`'" bigint comment '商家总成长值',
    "'`supplier_product_score`'" bigint comment '商家商品好评',
    "'`supplier_service_score`'" bigint comment '商家服务好评',
    "'`supplier_delivery_score`'" bigint comment '商家发货速度',
    "'`supplier_hot_sale_amount`'" bigint comment '商家热卖商品数量',
    "'`shop_type`'" STRING comment '店铺类型',
    "'`shop_name`'" STRING comment '店铺名称',
    "'`shop_add_time`'" STRING comment '店铺申请时间',
    "'`supplier_org_code`'" STRING comment '组织机构代码',
    "'`supplier_property`'" STRING comment '公司性质',
    "'`supplier_not_local`'" STRING comment '是否本地供应商',
    "'`shop_banner_title`'" STRING comment '店铺牌匾标题',
    "'`shop_seo_keyword`'" STRING comment '店铺SEO关键字',
    "'`supplier_week_sale_amount`'" bigint comment '商家周销量',
    "'`supplier_month_sale_amount`'" bigint comment '商家月销量',
    "'`supplier_product_hits`'" bigint comment '商家商品点击数',
    "'`shop_hits`'" bigint comment '店铺点击数',
    "'`supplier_codes`'" ARRAY<string> COMMENT '危化品资质代码',
    "'`supplier_area_id`'" STRING comment '商家所在地id',
    "'`supplier_station_id`'" STRING comment '商家所属配送区ID',
    "'`create_time`'" STRING comment '创建时间',
    "'`operate_time`'" STRING comment '操作时间',
    "'`start_date`'" STRING comment '开始日期',
    "'`end_date`'" STRING comment '结束日期'
)
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

#商家和店铺是一一对应的关系
supplier4="
with
    supplier_tmp as (SELECT id,"'`type`'",add_time,margin_id,company_name,county,shop_type,mode,security_account,cert_type,station_id,hot_sale_amount,
                            week_sale_amount,month_sale_amount,product_hits,product_score,service_score,delivery_score,org_code,property,not_local,grade_id,total_growth,
                            modified,created
                     FROM ods.ods_sys_tbl_supplier_upsert WHERE dt=$date_ods_first),
    supplier_shop_tmp as (SELECT "'`type`'",name,supplier_id,banner_title,seo_keyword,add_time,hits FROM ods.ods_sys_tbl_supplier_shop_upsert WHERE dt=$date_ods_first),
    supplier_danger_tmp as (SELECT supplier_id,collect_set(code) codes from ods.ods_sys_tbl_supplier_danger_cert_upsert WHERE dt=$date_ods_first GROUP by supplier_id)
insert overwrite table dim.dim_user_supplier_zipper partition(dt='99999999')
select
    supplier_tmp.id,
    supplier_tmp.add_time,
    supplier_tmp.company_name,
    margin_id,
    supplier_tmp."'`type`'",
    mode,
    cert_type,
    security_account,
    grade_id,
    total_growth,
    product_score,
    service_score,
    delivery_score,
    hot_sale_amount,
    supplier_shop_tmp."'`type`'",
    supplier_shop_tmp.name,
    supplier_shop_tmp.add_time,
    org_code,
    property,
    not_local,
    banner_title,
    seo_keyword,
    week_sale_amount,
    month_sale_amount,
    product_hits,
    hits ,
    codes,
    county,
    station_id,
    created,
    modified,
    $date_ods_first,
    '99999999'
from supplier_tmp
left join supplier_shop_tmp  on supplier_tmp.id=supplier_shop_tmp.supplier_id
left join supplier_danger_tmp  on supplier_tmp.id=supplier_danger_tmp.supplier_id;
"

product_create5="
DROP TABLE IF EXISTS dim.dim_product_product_zipper;
CREATE TABLE dim.dim_product_product_zipper(
                                           "'`product_id`'" STRING comment '商品id',
                                           "'`product_spu`'" STRING comment '商品spu(品牌+编码)',
                                           "'`product_name`'" STRING comment '商品名称',
                                           "'`product_code`'" STRING comment '商品编码',
                                           "'`product_cas`'" STRING comment '商品cas号',
                                           "'`product_supplier_id`'" STRING comment '商品所属商家ID',
                                           "'`product_mkt_price`'" DECIMAL(16,2) comment '商品市场价',
                                           "'`product_price`'" DECIMAL(16,2) comment '商品价格',
                                           "'`product_alert_amount`'" bigint comment '商品预警数量',
                                           "'`product_stock_amount`'" bigint comment '商品实际库存量',
                                           "'`product_usable_amount`'" bigint comment '商品可用库存量',
                                           "'`product_maker`'" STRING comment '商品生产企业',
                                           "'`product_bissness_type`'" STRING comment '商品业务类型',
                                           "'`product_sales_by_proxy`'" STRING comment '商品销售类型',
                                           "'`product_delivery_cycle`'" STRING comment '商品交货周期',
                                           "'`product_no_stock`'" STRING comment '商品是否零库存',
                                           "'`product_hot_sale_status`'" STRING comment '商品是否热卖',
                                           "'`product_type_id`'" STRING comment '商品所属类型ID',
                                           "'`product_brand_id`'" STRING comment '商品所属品牌ID',
                                           "'`product_cate_id`'" STRING comment '商品所属分类ID',
                                           "'`product_img`'" STRING comment '商品图片',
                                           "'`product_price_update_time`'" STRING comment '商品价格修改时间',
                                           "'`product_sale_time`'" STRING comment '商品上架时间',
                                           "'`product_instructions`'" STRING comment '商品是否具有说明书',
                                           "'`product_station_id`'" STRING comment '商品配送区id',
                                           "'`brand_name`'" STRING comment '品牌名称',
                                           "'`brand_recommend`'" STRING comment '是否推荐品牌',
                                           "'`brand_level`'" STRING comment '品牌代理级别',
                                           "'`cate_one_id`'" STRING comment '一级分类ID',
                                           "'`cate_one_name`'" STRING comment '一级分类名称',
                                           "'`cate_one_tax_name`'" STRING comment '一级税务分类名称',
                                           "'`cate_two_id`'" STRING comment '二级分类ID',
                                           "'`cate_two_name`'" STRING comment '二级分类名称',
                                           "'`cate_two_tax_name`'" STRING comment '二级税务分类名称',
                                           "'`cate_three_id`'" STRING comment '三级分类ID',
                                           "'`cate_three_name`'" STRING comment '三级分类名称',
                                           "'`cate_three_tax_name`'" STRING comment '三级税务分类名称',
                                           "'`cate_four_id`'" STRING comment '四级分类ID',
                                           "'`cate_four_name`'" STRING comment '四级分类名称',
                                           "'`cate_four_tax_name`'" STRING comment '四级税务分类名称',
                                           "'`cate_five_id`'" STRING comment '五级分类ID',
                                           "'`cate_five_name`'" STRING comment '五级分类名称',
                                           "'`cate_five_tax_name`'" STRING comment '五级税务分类名称',
                                           "'`type_name`'" STRING comment '类型名称',
                                           "'`tag_attrs`'" ARRAY<STRUCT<tag_id:bigint,name:STRING,weight:int>> comment '标签属性',
                                           "'`dangerous_name`'" STRING comment '危险品名称',
                                           "'`dangerous_sub_name`'" STRING comment '危险品别名',
                                           "'`dangerous_cate_name`'" STRING comment '危险品所属类目',
                                           "'`dangerous_description`'" STRING comment '危险品描述',
                                           "'`supplier_brand_agent_level`'" STRING comment '商家品牌代理级别',
                                           "'`supplier_brand_agent_area`'" STRING comment '商家品牌代理区域',
                                           "'`create_time`'" STRING comment '创建时间',
                                           "'`operate_time`'" STRING comment '操作时间',
                                           "'`start_date`'" STRING comment '开始日期',
                                           "'`end_date`'" STRING comment '结束日期'
)
PARTITIONED BY ("'`dt`'" STRING)
STORED AS PARQUET;
"

product5="
with
product_tmp as (SELECT id,
					code,
					cate_id,
					brand_id,
					type_id,
					supplier_id,
					name,
					mkt_price,
					price,
					temp_price,
					img,
					stock_amount,
					sale_amount,
					sale_time,
					alert_amount,
					usable_amount,
					no_stock,
					maker,
					sales_by_proxy,
					delivery_cycle,
					station_id,
					hot_sale_status,
					cas_code,
					bissness_type,
					created,
					modified,
					instructions_status,
					price_update_time
				FROM ods.ods_product_tbl_product_basic_upsert
				WHERE dt='$date_ods_first'),
brand_tmp as (SELECT id,
					name,
					recommend,
					reseller_level,
					main_id,
					mained
			FROM ods.ods_product_tbl_brand_upsert
			WHERE dt='$date_ods_first'),
cate_tmp0 as (SELECT id,
					split(CONCAT(pids,string(id)),':')[6] id5,
					split(CONCAT(pids,string(id)),':')[5] id4,
					split(CONCAT(pids,string(id)),':')[4] id3,
					split(CONCAT(pids,string(id)),':')[3] id2,
					split(CONCAT(pids,string(id)),':')[2] id1
			FROM ods.ods_product_tbl_category_full WHERE dt='$date_ods_first'),
cate_tmp1 as (SELECT id id1,name name1,tax_name tax_name1 FROM ods.ods_product_tbl_category_full WHERE dt='$date_ods_first'),
cate_tmp2 as (SELECT id id2,name name2,tax_name tax_name2 FROM ods.ods_product_tbl_category_full WHERE dt='$date_ods_first'),
cate_tmp3 as (SELECT id id3,name name3,tax_name tax_name3 FROM ods.ods_product_tbl_category_full WHERE dt='$date_ods_first'),
cate_tmp4 as (SELECT id id4,name name4,tax_name tax_name4 FROM ods.ods_product_tbl_category_full WHERE dt='$date_ods_first'),
cate_tmp5 as (SELECT id id5,name name5,tax_name tax_name5 FROM ods.ods_product_tbl_category_full WHERE dt='$date_ods_first'),
cate_tmp as (SELECT id,cate_tmp1.id1,name1,tax_name1,cate_tmp2.id2,name2,tax_name2,cate_tmp3.id3,name3,tax_name3,cate_tmp4.id4,name4,tax_name4,cate_tmp5.id5,name5,tax_name5
				FROM cate_tmp0
				left join cate_tmp1
				on cate_tmp0.id1=cate_tmp1.id1
				LEFT join cate_tmp2
				on cate_tmp0.id2=cate_tmp2.id2
				LEFT join cate_tmp3
				on cate_tmp0.id3=cate_tmp3.id3
				LEFT join cate_tmp4
				on cate_tmp0.id4=cate_tmp4.id4
				LEFT join cate_tmp5
				on cate_tmp0.id5=cate_tmp5.id5),
type_tmp as (SELECT id,name from ods.ods_product_tbl_product_type_full WHERE dt='$date_ods_first' ),
dangerous_tmp as (SELECT name,cas_code,sub_name,cate_name,description,sale_count from ods.ods_product_tbl_product_dangerous_full WHERE dt='$date_ods_first'),
agent_tmp as (SELECT supplier_id,brand_id,level,area
				from (
						SELECT supplier_id,brand_id,level,area ,row_number() over(PARTITION by supplier_id,brand_id ORDER by modified) rk
						from ods.ods_sys_tbl_supplier_brand_agent_upsert WHERE dt='$date_ods_first'
						) agent_tmp0
				WHERE agent_tmp0.rk=1),
tag_relt_tmp as (SELECT product_id,tag_id from ods.ods_product_tbl_product_relate_tag_upsert WHERE dt='$date_ods_first'),
tag_tmp as (SELECT id,name,weight from ods.ods_product_tbl_product_tag_upsert WHERE dt='$date_ods_first'),
prod_tag_tmp0 as (SELECT product_id,tag_id,name,weight FROM tag_relt_tmp LEFT JOIN tag_tmp on tag_relt_tmp.tag_id=tag_tmp.id),
prod_tag_tmp as (SELECT product_id,collect_set(named_struct('tag_id',tag_id,'name',name,'weight',weight)) tag_attrs
					FROM prod_tag_tmp0 group by product_id)
insert overwrite table dim.dim_product_product_zipper partition(dt='99999999')
select
	product_tmp.id product_id,
	CONCAT(CONCAT(product_tmp.brand_id,'_'),product_tmp.code) product_spu,
	product_tmp.name product_name,
	product_tmp.code product_code,
	product_tmp.cas_code cas_code,
	product_tmp.supplier_id product_supplier_id,
	mkt_price product_mkt_price,
	product_tmp.price product_price,
	alert_amount product_alert_amount,
	stock_amount product_stock_amount,
	usable_amount product_usable_amount,
	product_tmp.maker product_maker,
	bissness_type product_bissness_type,
	sales_by_proxy product_sales_by_proxy,
	delivery_cycle product_delivery_cycle,
	no_stock product_no_stock,
	hot_sale_status product_hot_sale_status,
	product_tmp.type_id product_type_id,
	product_tmp.brand_id product_brand_id,
	product_tmp.cate_id product_cate_id,
	product_tmp.img product_img,
	price_update_time product_price_update_time,
	product_tmp.sale_time product_sale_time,
	instructions_status product_instructions,
	product_tmp.station_id product_station_id,
	brand_tmp.name brand_name,
	brand_tmp.recommend brand_recommend,
	brand_tmp.reseller_level brand_level,
	id1,
	name1,
	tax_name1,
	id2,
	name2,
	tax_name2,
	id3,
	name3,
	tax_name3,
	id4,
	name4,
	tax_name4,
	id5,
	name5,
	tax_name5,
	type_tmp.name type_name,
	tag_attrs,
	dangerous_tmp.name dangerous_name,
	dangerous_tmp.sub_name dangerous_sub_name,
	dangerous_tmp.cate_name dangerous_cate_name,
	dangerous_tmp.description dangerous_description,
	agent_tmp.level supplier_brand_agent_level,
	agent_tmp.area supplier_brand_agent_area,
	product_tmp.created,
	product_tmp.modified,
    '$date_ods_first',
    '99999999'
    FROM product_tmp
    left join brand_tmp
    on product_tmp.brand_id=brand_tmp.id
    LEFT JOIN cate_tmp
    on product_tmp.cate_id=cate_tmp.id
	LEFT JOIN type_tmp
	on product_tmp.type_id=type_tmp.id
	LEFT JOIN prod_tag_tmp
	on product_tmp.id=prod_tag_tmp.product_id
	left join dangerous_tmp
	on product_tmp.cas_code=dangerous_tmp.cas_code
	LEFT JOIN agent_tmp
	on product_tmp.supplier_id=agent_tmp.supplier_id and product_tmp.brand_id=agent_tmp.brand_id;
"

case $1 in
    member )
        hive -e "$member_create3"
        hive -e "$member3"
    ;;
     supplier )
        hive -e "$supplier_create4"
        hive -e "$supplier4"
    ;;
    all )
        hive -e "$area_create1"
        hive -e "$station_create2"
        hive -e "$member_create3"
        hive -e "$supplier_create4"
        hive -e "$product_create5"
        hive -e "$area1"
        hive -e "$station2"
        hive -e "$member3"
        hive -e "$supplier4"
        hive -e "$product5"
    ;;
esac






















