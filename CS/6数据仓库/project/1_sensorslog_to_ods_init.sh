#!/bin/sh
#手动执行该脚本，需要切换dolphinscheduler用户，才有ssh神策权限。调度时，自动使用dolphinscheduler用户。
#该脚本全程使用dolphinscheduler用户执行，应该没有用户权限的问题。

#首次加载，创建CDH中的ods层的日志hive表
hive -e "drop table if exists ods.ods_sensorslog_events;"
hive -e 'CREATE TABLE ods.ods_sensorslog_events (event STRING,user_id BIGINT,distinct_id STRING,`date` TIMESTAMP,`time` TIMESTAMP,is_login BIGINT,user_type STRING,receive_region STRING,current_page STRING,search_type STRING,search_limit STRING,key_word_type STRING,key_word STRING,source_page STRING,referrerbanner_id STRING,referrerbanner_name STRING,good_id STRING,good_name STRING,specification STRING,brand STRING,brand_agent_level STRING,brand_agent_district STRING,first_good STRING,second_good STRING,third_good STRING,good_price DOUBLE,store_id STRING,store_name STRING,supplier_id STRING,supplier_name STRING,source_key_word STRING,search_content STRING,is_request BIGINT,result_number DOUBLE,transportation_costs DOUBLE,good_quantity DOUBLE,discount DOUBLE,order_id STRING,receiver_area STRING,account_type STRING,receiver_province STRING,receiver_city STRING,good_type DOUBLE,is_coupon BIGINT,order_amount DOUBLE,receiver_name STRING,receiver_address STRING,pay_type STRING,invoice_type STRING,is_success BIGINT,fail_reason STRING,account STRING,shoppingcart_entrance STRING,good_id_list STRING,good_name_list STRING,good_price_list STRING,store_id_list STRING,store_name_list STRING,supplier_id_list STRING,supplier_name_list STRING,banner_belong_area STRING,banner_type STRING,banner_name STRING,banner_id STRING,banner_rank DOUBLE,position_number STRING,false_result STRING,collect_type STRING,platform STRING,activity_name STRING,activity_id STRING,activity_states STRING,activity_type STRING,main_venue DOUBLE,parallel_session DOUBLE,venue_id STRING,venue_name STRING,on_main_venue STRING,location_id STRING,location_name STRING,good_manage_number STRING,good_sort STRING,good_code STRING,location_botton_name STRING,next_page STRING,cate STRING,recommender_position STRING,recommender_name STRING,login_type STRING,platform_type STRING,button_name STRING,site_id STRING,screen_height DOUBLE,screen_width DOUBLE,lib STRING,latest_traffic_source_type STRING,latest_search_keyword STRING,latest_referrer STRING,referrer STRING,url STRING,url_path STRING,title STRING,is_first_day BIGINT,is_first_time BIGINT,referrer_host STRING,ip STRING,url_host STRING,os STRING,os_version STRING,browser STRING,browser_version STRING,track_signup_original_id STRING,idmap_reason STRING,city STRING,province STRING,country STRING,manufacturer STRING,model STRING,receive_time DOUBLE,element_type STRING,element_name STRING,element_class_name STRING,element_content STRING,viewport_width DOUBLE,element_selector STRING,bot_name STRING,latest_utm_campaign STRING,latest_utm_source STRING,element_target_url STRING,element_id STRING,viewport_position DOUBLE,viewport_height DOUBLE,event_duration DOUBLE,latest_utm_medium STRING,timezone_offset DOUBLE,element_path STRING,latest_referrer_host STRING,latest_utm_content STRING,latest_utm_term STRING) PARTITIONED BY (dt string) STORED AS PARQUET;'

# 每次同步一定时间范围内的数据数据
# 传入两个参数，开始时间，结束时间，格式形如 2020-01-01
# 数据日期包含开始时间，结束时间
import_range_data(){
  #远程登录到神策数据库,执行以下命令
  #删除上次遗留的神策本地文件
  #将要导出的数创建一张hive表
  #将hive表hdfs文件复制到神策本地
  #删除神策中我们创建的hive表
  ssh -p36910 zhanghongyang@47.111.170.130 << eeooff
rm -rf /home/zhanghongyang/export_data
impala-shell -q 'drop table if exists default.export_data;'
impala-shell -q 'CREATE TABLE export_data stored AS parquet location "/tmp/impala/export_data" AS /*SA_BEGIN(production)*/ SELECT event,user_id,distinct_id,date,time,is_login,user_type,receive_region,current_page,search_type,search_limit,key_word_type,key_word,source_page,referrerbanner_id,referrerbanner_name,good_id,good_name,specification,brand,brand_agent_level,brand_agent_district,first_good,second_good,third_good,good_price,store_id,store_name,supplier_id,supplier_name,source_key_word,search_content,is_request,result_number,transportation_costs,good_quantity,discount,order_id,receiver_area,account_type,receiver_province,receiver_city,good_type,is_coupon,order_amount,receiver_name,receiver_address,pay_type,invoice_type,is_success,fail_reason,account,shoppingcart_entrance,good_id_list,good_name_list,good_price_list,store_id_list,store_name_list,supplier_id_list,supplier_name_list,banner_belong_area,banner_type,banner_name,banner_id,banner_rank,position_number,false_result,collect_type,platform,activity_name,activity_id,activity_states,activity_type,main_venue,parallel_session,venue_id,venue_name,on_main_venue,location_id,location_name,good_manage_number,good_sort,good_code,location_botton_name,next_page,cate,recommender_position,recommender_name,login_type,platform_type,button_name,site_id,''\$screen_height'' AS screen_height,''\$screen_width'' AS screen_width,''\$lib'' AS lib,''\$latest_traffic_source_type'' AS latest_traffic_source_type,''\$latest_search_keyword'' AS latest_search_keyword,''\$latest_referrer'' AS latest_referrer,''\$referrer'' AS referrer,''\$url'' AS url,''\$url_path'' AS url_path,''\$title'' AS title,''\$is_first_day'' AS is_first_day,''\$is_first_time'' AS is_first_time,''\$referrer_host'' AS referrer_host,''\$ip'' AS ip,''\$url_host'' AS url_host,''\$os'' AS os,''\$os_version'' AS os_version,''\$browser'' AS browser,''\$browser_version'' AS browser_version,''\$track_signup_original_id'' AS track_signup_original_id,''\$idmap_reason'' AS idmap_reason,''\$city'' AS city,''\$province'' AS province,''\$country'' AS country,''\$manufacturer'' AS manufacturer,''\$model'' AS model,''\$receive_time'' AS receive_time,''\$element_type'' AS element_type,''\$element_name'' AS element_name,''\$element_class_name'' AS element_class_name,''\$element_content'' AS element_content,''\$viewport_width'' AS viewport_width,''\$element_selector'' AS element_selector,''\$bot_name'' AS bot_name,''\$latest_utm_campaign'' AS latest_utm_campaign,''\$latest_utm_source'' AS latest_utm_source,''\$element_target_url'' AS element_target_url,''\$element_id'' AS element_id,''\$viewport_position'' AS viewport_position,''\$viewport_height'' AS viewport_height,''\$event_duration'' AS event_duration,''\$latest_utm_medium'' AS latest_utm_medium,''\$timezone_offset'' AS timezone_offset,''\$element_path'' AS element_path,''\$latest_referrer_host'' AS latest_referrer_host,''\$latest_utm_content'' AS latest_utm_content,''\$latest_utm_term'' AS latest_utm_term FROM events where date>='"'$1'"' and date<='"'$2'"' /*SA_END*/;'
hadoop fs -get /tmp/impala/export_data /home/zhanghongyang/
impala-shell -q 'drop table export_data'
exit
eeooff
  #复制文件从神策本地到CDH本地
  rm -rf /home/dolphinscheduler/log_data
  scp -r -P36910 zhanghongyang@47.111.170.130:/home/zhanghongyang/export_data/ /home/dolphinscheduler/log_data
  #将本地文件移动到HDFS
  hadoop fs -rm -r /tmp/hive/sensorslog/log_data
  hadoop fs -moveFromLocal /home/dolphinscheduler/log_data /tmp/hive/sensorslog/
  #将数据加载进hive表
  do_dt=$2
  do_dt=${do_dt//-/}
  hive -e "load data inpath '/tmp/hive/sensorslog/log_data' into table ods.ods_sensorslog_events partition(dt=$do_dt);"
}



#生成要导入数据的日期
if [ -n "$1" ] ;then
    do_date=$1
else
    do_date=`date -d '-1 day' +%F`
fi



#生成要导入数据的日期,暂不生成，手动写入
import_range_data 2021-07-01 2021-07-31
import_range_data 2021-08-01 2021-08-31
import_range_data 2021-09-01 2021-09-30
import_range_data 2021-10-01 2021-10-31
import_range_data 2021-11-01 2021-11-30
import_range_data 2021-12-01 2021-12-31
import_range_data 2022-01-01 2022-01-31
import_range_data 2022-02-01 2022-02-17