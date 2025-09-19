#!/bin/sh
#业务数据库首日导入脚本
#使用示例：sh import_casmart_product_init_parallel.sh all 20200216

#定义一些需要的变量
sqoop=/opt/cloudera/parcels/CDH-6.3.2-1.cdh6.3.2.p0.1605554/bin/sqoop
mysql_url=rr-2zecu7ibaj063n166.mysql.rds.aliyuncs.com
mysql_username=zhanghongyang
mysql_password=1_hzMEi5uxepfbjpmh

#传入的日期参数 格式形如 20210614
if [ -n "$2" ] ;then
    do_date=$2
else
    do_date=`date -d '-1 day' +%F`
    do_date=${do_date//-/}
fi

#使用sqoop复制casmart业务库表结构 创建hive ods层表结构
#第一个参数是业务数据库名称，第二个参数是业务表名称,第三个参数是Hive中表名称
create_hive_ods_table(){
  sqoop create-hive-table \
  -D mapred.job.name="job name: create table [$3]" \
  --connect jdbc:mysql://$mysql_url/$1 \
  --username $mysql_username \
  --password "$mysql_password" \
  --table $2 \
  --hive-database 'ods' \
  --hive-table $3 \
  --hive-overwrite \
  --hive-partition-key 'dt'
  }


#把数据从mysql导入到hdfs并load进hive
#该函数第一个参数代表业务数据库名称，第二个参数是业务表名称,第三个参数是sql语句,第四个参数是Hive中表名称,第5个参数是切分表的列，实现多map任务同时执行
#这种导入方式会覆盖原有数据
import_data(){
  $sqoop import \
  -D mapred.job.name="job name: import table [$2]" \
  --connect jdbc:mysql://$mysql_url/$1 \
  --username $mysql_username \
  --password "$mysql_password" \
  --query "$3 where \$CONDITIONS" \
  --target-dir /casmart/$1/$2/$do_date \
  --delete-target-dir \
  --null-string '\\N' \
  --null-non-string '\\N' \
  --num-mappers $6 \
  --split-by "$5" \
  --compress \
  --compression-codec SnappyCodec \
  --hive-import \
  --hive-overwrite \
  --hive-database 'ods' \
  --hive-table $4 \
  --hive-partition-key 'dt' \
  --hive-partition-value $do_date \
  --hive-drop-import-delims
  }

#这种导入方式会追加数据
import_data_add(){
  $sqoop import \
  -D mapred.job.name="job name: import table [$2]" \
  --connect jdbc:mysql://$mysql_url/$1 \
  --username $mysql_username \
  --password "$mysql_password" \
  --query "$3 where \$CONDITIONS" \
  --target-dir /casmart/$1/$2/$do_date \
  --delete-target-dir \
  --null-string '\\N' \
  --null-non-string '\\N' \
  --num-mappers $6 \
  --split-by "$5" \
  --compress \
  --compression-codec SnappyCodec \
  --hive-import \
  --hive-database 'ods' \
  --hive-table $4 \
  --hive-partition-key 'dt' \
  --hive-partition-value $do_date \
  --hive-drop-import-delims
  }

#首次全量同步mysql表
import_ods_product_tbl_brand(){
  create_hive_ods_table db_casmart_product2 tbl_brand ods_product_tbl_brand_upsert
	import_data db_casmart_product2 tbl_brand "select * from tbl_brand" ods_product_tbl_brand_upsert id 1
	}

import_ods_product_tbl_category(){
  create_hive_ods_table db_casmart_product2 tbl_category ods_product_tbl_category_full
	import_data db_casmart_product2 tbl_category "select * from tbl_category" ods_product_tbl_category_full id 1
	}

import_ods_product_tbl_product_type(){
  create_hive_ods_table db_casmart_product2 tbl_product_type ods_product_tbl_product_type_full
	import_data db_casmart_product2 tbl_product_type "select * from tbl_product_type" ods_product_tbl_product_type_full id 1
	}

import_ods_product_tbl_product_relate_tag(){
  create_hive_ods_table db_casmart_product2 tbl_product_relate_tag ods_product_tbl_product_relate_tag_upsert
	import_data db_casmart_product2 tbl_product_relate_tag "select * from tbl_product_relate_tag" ods_product_tbl_product_relate_tag_upsert id 1
	}

import_ods_product_tbl_sys_material(){
  create_hive_ods_table db_casmart_product2 tbl_sys_material ods_product_tbl_sys_material_once
	import_data db_casmart_product2 tbl_sys_material "select * from tbl_sys_material" ods_product_tbl_sys_material_once id 1
	}

import_ods_product_tbl_product_dangerous(){
  create_hive_ods_table db_casmart_product2 tbl_product_dangerous ods_product_tbl_product_dangerous_full
	import_data db_casmart_product2 tbl_product_dangerous "select * from tbl_product_dangerous" ods_product_tbl_product_dangerous_full id 1
	}

import_ods_product_tbl_product_tag(){
  create_hive_ods_table db_casmart_product2 tbl_product_tag ods_product_tbl_product_tag_upsert
	import_data db_casmart_product2 tbl_product_tag "select * from tbl_product_tag" ods_product_tbl_product_tag_upsert id 1
	}

import_ods_product_tbl_nine_product(){
  create_hive_ods_table db_casmart_product2 tbl_nine_product ods_product_tbl_nine_product_upsert
  import_data db_casmart_product2 tbl_nine_product "select * from tbl_nine_product" ods_product_tbl_nine_product_upsert id 1
}

import_ods_product_tbl_product_intro(){
  create_hive_ods_table db_casmart_product2 tbl_product_intro ods_product_tbl_product_intro_upsert
  import_data db_casmart_product2 tbl_product_intro "select * from tbl_product_intro" ods_product_tbl_product_intro_upsert id 1
}

import_ods_product_tbl_product_price_report(){
  create_hive_ods_table db_casmart_product2 tbl_product_price_report ods_product_tbl_product_price_report_full
  import_data db_casmart_product2 tbl_product_price_report "select * from tbl_product_price_report" ods_product_tbl_product_price_report_full id 1
}

import_ods_product_tbl_product_correction(){
  create_hive_ods_table db_casmart_product2 tbl_product_correction ods_product_tbl_product_correction_full
  import_data db_casmart_product2 tbl_product_correction "select * from tbl_product_correction" ods_product_tbl_product_correction_full id 1
}

import_ods_product_tbl_product_basic(){
  create_hive_ods_table db_casmart_product2 tbl_product_basic_0 ods_product_tbl_product_basic_upsert
  import_data db_casmart_product2 tbl_product_basic_0 "select * from tbl_product_basic_0" ods_product_tbl_product_basic_upsert id 1
  for((i=1;i<=100;i++));
  do
    import_data_add db_casmart_product2 tbl_product_basic_$i "select * from tbl_product_basic_$i" ods_product_tbl_product_basic_upsert id 1
  done
  }

#传入不同参数，即执行本脚本不同函数
case $1 in
  ("tbl_brand")
    import_ods_product_tbl_brand
;;
  ("tbl_category")
    import_ods_product_tbl_category
;;
  ("tbl_product_type")
    import_ods_product_tbl_product_type
;;
  ("tbl_product_relate_tag")
    import_ods_product_tbl_product_relate_tag
;;
  ("tbl_sys_material")
    import_ods_product_tbl_sys_material
;;
  ("tbl_product_dangerous")
    import_ods_product_tbl_product_dangerous
;;
  ("tbl_product_tag")
    import_ods_product_tbl_product_tag
;;
  ("tbl_nine_product")
    import_ods_product_tbl_nine_product
;;
  ("tbl_product_intro")
    import_ods_product_tbl_product_intro
;;
  ("tbl_product_price_report")
    import_ods_product_tbl_product_price_report
;;
  ("tbl_product_correction")
    import_ods_product_tbl_product_correction
;;
  ("tbl_product_basic")
    import_ods_product_tbl_product_basic
;;
  ("all")
    import_ods_product_tbl_brand
    import_ods_product_tbl_category
    import_ods_product_tbl_product_type
    import_ods_product_tbl_product_relate_tag
    import_ods_product_tbl_sys_material
    import_ods_product_tbl_product_dangerous
    import_ods_product_tbl_product_tag
    import_ods_product_tbl_nine_product
    import_ods_product_tbl_product_intro
    import_ods_product_tbl_product_price_report
    import_ods_product_tbl_product_correction
    import_ods_product_tbl_product_basic
;;
esac