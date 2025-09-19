#!/bin/sh
#手动执行该脚本，需要切换dolphinscheduler用户，才有ssh神策权限。调度时，自动使用dolphinscheduler用户。
#该脚本全程使用dolphinscheduler用户执行，应该没有用户权限的问题。
#从20220218开始执行每日脚本


# 每次同步一定时间范围内的数据数据
# 传入两个参数，开始时间，结束时间，格式形如 2020-01-01
# 数据日期包含开始时间，结束时间
import_oneday_data(){
  #远程登录到神策数据库,执行以下命令
  #删除上次遗留的神策本地文件和表
  #将要导出的数创建一张hive表
  #将hive表hdfs文件复制到神策本地
  #删除神策中我们创建的hive表
  ssh -p36910 zhanghongyang@47.111.170.130 << eeooff
rm -rf /home/zhanghongyang/export_data
impala-shell -q 'drop table if exists default.export_data;'
impala-shell -q 'CREATE TABLE export_data stored AS parquet location "/tmp/impala/export_data" AS /*SA_BEGIN(production)*/ SELECT event,user_id,distinct_id,date,time,is_login,user_type,receive_region,current_page,search_type,search_limit,key_word_type,key_word,source_page,referrerbanner_id,referrerbanner_name,good_id,good_name,specification,brand,brand_agent_level,brand_agent_district,first_good,second_good,third_good,good_price,store_id,store_name,supplier_id,supplier_name,source_key_word,search_content,is_request,result_number,transportation_costs,good_quantity,discount,order_id,receiver_area,account_type,receiver_province,receiver_city,good_type,is_coupon,order_amount,receiver_name,receiver_address,pay_type,invoice_type,is_success,fail_reason,account,shoppingcart_entrance,good_id_list,good_name_list,good_price_list,store_id_list,store_name_list,supplier_id_list,supplier_name_list,banner_belong_area,banner_type,banner_name,banner_id,banner_rank,position_number,false_result,collect_type,platform,activity_name,activity_id,activity_states,activity_type,main_venue,parallel_session,venue_id,venue_name,on_main_venue,location_id,location_name,good_manage_number,good_sort,good_code,location_botton_name,next_page,cate,recommender_position,recommender_name,login_type,platform_type,button_name,site_id,''\$screen_height'' AS screen_height,''\$screen_width'' AS screen_width,''\$lib'' AS lib,''\$latest_traffic_source_type'' AS latest_traffic_source_type,''\$latest_search_keyword'' AS latest_search_keyword,''\$latest_referrer'' AS latest_referrer,''\$referrer'' AS referrer,''\$url'' AS url,''\$url_path'' AS url_path,''\$title'' AS title,''\$is_first_day'' AS is_first_day,''\$is_first_time'' AS is_first_time,''\$referrer_host'' AS referrer_host,''\$ip'' AS ip,''\$url_host'' AS url_host,''\$os'' AS os,''\$os_version'' AS os_version,''\$browser'' AS browser,''\$browser_version'' AS browser_version,''\$track_signup_original_id'' AS track_signup_original_id,''\$idmap_reason'' AS idmap_reason,''\$city'' AS city,''\$province'' AS province,''\$country'' AS country,''\$manufacturer'' AS manufacturer,''\$model'' AS model,''\$receive_time'' AS receive_time,''\$element_type'' AS element_type,''\$element_name'' AS element_name,''\$element_class_name'' AS element_class_name,''\$element_content'' AS element_content,''\$viewport_width'' AS viewport_width,''\$element_selector'' AS element_selector,''\$bot_name'' AS bot_name,''\$latest_utm_campaign'' AS latest_utm_campaign,''\$latest_utm_source'' AS latest_utm_source,''\$element_target_url'' AS element_target_url,''\$element_id'' AS element_id,''\$viewport_position'' AS viewport_position,''\$viewport_height'' AS viewport_height,''\$event_duration'' AS event_duration,''\$latest_utm_medium'' AS latest_utm_medium,''\$timezone_offset'' AS timezone_offset,''\$element_path'' AS element_path,''\$latest_referrer_host'' AS latest_referrer_host,''\$latest_utm_content'' AS latest_utm_content,''\$latest_utm_term'' AS latest_utm_term FROM events where date='"'$1'"' /*SA_END*/;'
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
  do_date=$1
  do_date=${do_date//-/}
  hive -e "load data inpath '/tmp/hive/sensorslog/log_data' into table ods.ods_sensorslog_events partition(dt=$do_date);"
}


#生成要导入数据的日期
if [ -n "$1" ] ;then
    do_date=$1
else
    do_date=`date -d '-1 day' +%F`
fi

#导入数据
import_oneday_data $do_date
















