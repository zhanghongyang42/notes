# 发展历程

SQL(Structure Query Language)语言是数据库的核心语言。数据控制语言DCL是SQL语言四大主要分类之一。



SQL的发展是从1974年开始的，其发展过程如下：
		1974年—–由Boyce和Chamberlin提出，当时称SEQUEL。
		1976年—–IBM公司的Sanjase研究所在研制RDBMS SYSTEM R时改为SQL。
		1979年—–ORACLE公司发表第一个基于SQL的商业化RDBMS产品。
		1982年—–IBM公司出版第一个RDBMS语言SQL/DS。
		1985年—–IBM公司出版第一个RDBMS语言DB2。
		1986年—–美国国家标准化组织ANSI宣布SQL作为数据库工业标准。



SQL是一个标准的数据库语言，是面向集合的描述性非过程化语言。

非过程性语言，即大多数语句都是独立执行的，与上下文无关，而绝大部分应用都是一个完整的过程，显然用SQL完全实现这些功能是很困难的。

为了解决此问题，作了如下两方面的工作：

​		1、扩充SQL，在SQL中引入过程性结构；

​		2、把SQL嵌入到高级语言中，以便一起完成一个完整的应用。



# DDL(数据定义语言)

### 数据库

创建数据库            		create database 数据库名;

使用数据库            		use 数据库名;

删除数据库            		drop database 数据库名;

查看所有数据库            show databases;

选择数据库   				 selct datebada();



### 数据表

创建表          

```sql
create table 表名(
    列名 数据类型 约束,
    列名 数据类型 约束);
```

  

查看所有表     		 		show tables;

查看表结构					  desc 表名;

复制表							 create table 表名 like 被复制的表名;



删除表         

```sql
drop table 表名;   
delete from 表名;   
truncate table 表名;
```

​            		

修改表

```sql
-- 添加列名称			
alter table 表名 add 列名 数据类型 约束;
-- 修改列名称		
alter table 表名 change 旧列名 新列名 类型 约束;
-- 修改列的数据类型约束	
alter table 表名 modify 列名 数据类型 约束;
-- 删除列					
alter table 表名 drop列名;
```



# DML(数据操作语言)

添加数据      

```sql
intert into 表名 values(全列值),(全列值); 
intert into 表名(列1,列2)values(值1,值2),(值1,值2);
```

修改数据

```sql
update 表名 set 列名 = 值,列名 = 值 where 条件
```

删除数据

```sql
delete from 表名 where 条件; 
```



# DCL (数据控制语言)

GRANT：此命令用于把语句权限或者对象权限授予给其他用户和角色。

```
GRANT {ALL | statement[,...n]} TO security_account [ ,...n ]
```



DENY：此命令用于拒绝给当前数据库内的用户或者角色授予权限，并防止用户或角色通过其组或角色成员继承权限。

```
DENY { ALL | statement [ ,...n ] } TO security_account [ ,...n ]
```



REVOKE：REVOKE语句是与GRANT语句相反的语句，它能够将以前在当前数据库内的用户或者角色上授予或拒绝的权限删除，但是该语句并不影响用户或者角色从其他角色中作为成员继承过来的权限。

```
REVOKE { ALL | statement [ ,...n ] } FROM security_account [ ,...n ]
```



# DQL(数据查询语言)

```sql
select [distinct]列名 [as 新名字]
		from 表名 where 条件 
		group by 分组字段 having 分组后条件 
		order by 排序字段 [asc]/desc
```

执行顺序		from--where--group by--having--select--order by

分组后，select只能查分组字段，或者其他字段的聚合。

查询的时候要注意null值进行运算还是null，会被where后的运算过滤掉。

子查询的子表要起别名，不然报错。



### 运算符

```
+-*/
=、>、<、>=、<=、<>
AND、OR、NOT
```

查询的时候要注意null值进行运算还是null，会被where后的运算过滤掉



### 关键字

```sql
SELECT * FROM Persons LIMIT 0,5;
SELECT * FROM Persons WHERE LastName IN ('Adams','Carter');
SELECT LastName,FirstName,Address FROM Persons WHERE Address IS NULL;
SELECT * FROM Persons WHERE LastName [NOT] BETWEEN 'Adams' AND 'Carter';
SELECT * FROM Persons WHERE City LIKE '%N%'
```



### SQL连接

左右连接时，从表字段的值如果不唯一，会使数据量增多

```sql
-- 笛卡尔积
SELECT * FROM Persons INNER JOIN Orders
-- 内连接
SELECT * FROM Persons Orders ON Persons.Id_P = Orders.Id_P
SELECT * FROM Persons INNER JOIN Orders ON Persons.Id_P = Orders.Id_P
-- 左外链接,左表数据全保留，条件筛选右表
select * from a left outer join b on 条件;   
-- 全连接，两表能连接的连接，不能的保留
select * from a full join b on 条件;    
```



### 集合运算

```sql
-- 并集
(SELECT * FROM instructor WHERE name='smith')
UNION [all]
(SELECT * FROM instructor WHERE dept_name = 'history');

-- 交集
(SELECT dept_name FROM instructor WHERE name = 'smith')
intersect
(SELECT dept_name FROM department);

--差集
(SELECT dept_name FROM instructor)
EXCEPT 
(SELECT dept_name FROM department WHERE dept_name = 'biology');

-- mysql交集
SELECT DISTINCT dept_name FROM 
instructor
INNER JOIN 
department 
on instructor.dept_name=department.dept_name

-- mysql差集
SELECT dept_name FROM 
department
LEFT JOIN instructor 
on instructor.dept_name=department.dept_name
WHERE instructor.dept_name IS NULL ;
```



# 函数

### case表达式

```sql
-- case end 是用if，else筛选name值。when then是顺序执行的，所以要注意条件的顺序，上面满足了，下面就不会执行了。
SELECT name
  , CASE 
    WHEN math_score >= 80
    AND eng_score >= 80 THEN '优'
    WHEN math_score >= 60
    AND eng_score >= 60 THEN '良'
    WHEN math_score >= 60
    OR eng_score >= 60 THEN '中'
    WHEN math_score <= 60
    AND eng_score < 60 THEN '差'
    ELSE NULL
  END AS score_grade
FROM student
```



### 日期函数

```
now()         	返回当前的日期和时间
curdate()     	返回当前的日期
curtime()     	返回当前的时间
 
date()        	提取日期或日期/时间表达式的日期部分
extract()     	返回日期/时间按的单独部分
 
date_add()    	给日期添加指定的时间间隔
date_sub()    	从日期减去指定的时间间隔
 
datediff()    	返回两个日期之间的天数
date_format()  	用不同的格式显示日期/时间
```



###  合计函数

```
avg(column)           返回某列的平均值
count(column)         返回某列的行数（不包括 null 值）
count(*)                             返回被选行数
first(column)         返回在指定的域中第一个记录的值
last(column)          返回在指定的域中最后一个记录的值
max(column)           返回某列的最高值
min(column)           返回某列的最低值
stdev(column)          
stdevp(column)         
sum(column)           返回某列的总和
var(column)            
varp(column)           
```



### Scalar 函数

```
ucase(c)                      	将某个域转换为大写
lcase(c)                       	将某个域转换为小写
mid(c,start[,end])              从某个文本域提取字符
len(c)                          返回某个文本域的长度
instr(c,char)                  	返回在某个文本域中指定字符的数值位置
left(c,number_of_char)       	返回某个被请求的文本域的左侧部分
right(c,number_of_char)     	返回某个被请求的文本域的右侧部分
round(c,decimals)             	对某个数值域进行指定小数位数的四舍五入
mod(x,y)                     	返回除法操作的余数
datediff(d,date1,date2)       	用于执行日期计算
format(c,format)              	改变某个域的显示方式
```



# 约束  

```sql
CREATE TABLE Persons(
    Id_P int CHECK (Id_P>0) AUTO_INCREMENT, -- 自增
    LastName varchar(255) NOT NULL,  -- 非空
    FirstName varchar(255) UNIQUE,  -- 唯一
    Address varchar(255) PRIMARY KEY , -- 主键:非空唯一
    City varchar(255) DEFAULT 'Sandnes', -- 默认值
    Id_P int FOREIGN KEY REFERENCES Orders(Id_O) --外键
)
```



```sql
-- 添加约束
ALTER TABLE Persons ADD UNIQUE (Id_P)
-- 删除约束
ALTER TABLE Persons DROP CONSTRAINT Id_P
```



**外键**

```
一个表的外键，一定指向另一个表的主键，两个表的主外键 列相同。
外键值只能是空值，或者是主键中出现的值。
```



# 索引

常用索引原理：B+ 树



**建立原则**

1. 更新频繁的列不应设置索引
2. 数据量小的表不要使用索引（毕竟总共2页的文档，还要目录吗？）
3. 重复数据多的字段不应设为索引（比如性别，只有男和女，一般来说：重复的数据超过百分之十五就不适合建索引）
4. 首先应该考虑对where 和 order by 使用的列上建立索引



创建

```sql
--建表时
CREATE TABLE 表名(
字段名 数据类型 [完整性约束条件],
       ……，
[UNIQUE | FULLTEXT | SPATIAL] INDEX
[索引名](字段名1  [ASC | DESC])
);

--建表后
ALTER TABLE 表名 ADD [UNIQUE| FULLTEXT | SPATIAL]  INDEX | KEY  [索引名] (字段名1 [(长度)] [ASC | DESC]) [USING 索引方法]；
 
--说明：
--UNIQUE:可选。表示索引为唯一性索引。
--FULLTEXT:可选。表示索引为全文索引。
--SPATIAL:可选。表示索引为空间索引。
--INDEX:用于指定字段为索引。
--索引名:可选。给创建的索引取一个新名称。
--字段名1:指定索引对应的字段的名称，该字段必须是前面定义好的字段。

--示例
CREATE TABLE classInfo(
    id INT AUTO_INCREMENT COMMENT 'id',
    classname VARCHAR(128) COMMENT '课程名称',
    classid INT COMMENT '课程id',
    classtype VARCHAR(128) COMMENT '课程类型',
    classcode VARCHAR(128) COMMENT '课程代码',
-- 主键本身也是一种索引
    PRIMARY KEY (id),
-- 给classid字段创建了唯一索引(注:也可以在上面创建字段时使用unique来创建唯一索引)
    UNIQUE INDEX (classid),
-- 给classname字段创建普通索引
    INDEX (classname),
-- 创建组合索引
    INDEX (classtype,classcode)
);

ALTER TABLE classInfo ADD UNIQUE INDEX (classid);
```

 删查

```sql
drop index classname on classInfo;
show index from classInfo;
```



# 视图

数据库中的数据都是存储在表中的，而视图只是一个或多个表依照某个条件组合而成的结果集，一般来说视图只能进行select操作。

但是也存在可更新的视图，对于这类视图的update，insert和delete等操作最终会作用于与其相关的表中数据。

表是数据库中数据存储的基础，而视图只是为了满足某种查询要求而建立的一个对象。



使用场景：

1. 视图是数据库数据的特定子集。可以禁止所有用户访问数据库表，而要求用户只能通过视图操作数据，这种方法可以保护数据库不受用户和应用程序的影响
2. 视图有时会对提高效率有帮助。临时表几乎是不会对性能有帮助，是资源消耗者。 视图一般存放在该数据库，临时表永远都是在tempdb里的。 



#  表关系

一对一，一对多，使用外键即可。

多对多，使用中间表。



# 存储过程

存储过程（Stored Procedure）是在大型数据库系统中，一组为了完成特定功能的SQL 语句集，存储在数据库中，经过第一次编译后再次调用不需要再次编译，用户通过指定存储过程的名字并给出参数（如果该存储过程带有参数）来调用存储过程。

可以由数据库自己去调用，也可以由java程序去调用。



创建及调用：https://www.cnblogs.com/geaozhang/p/6797357.html



# SQL 分析

### explain

可以返回字段供我们分析查询速度：

```sql
explain select * from user_info where id = 2
explain format=json select * from user_info where id = 2
```



Explain命令返回的信息中包括以下几个重要的字段：

- id: 执行计划中每个操作的唯一标识符。
- select_type: 操作类型（例如SIMPLE、PRIMARY、SUBQUERY、UNION、PRIMARY-UNION、DEPENDENT UNION等）。
- table: 操作表名。
- partitions: 操作的分区。
- type: 操作类型（例如system、const、eq_ref、ref、range、index、all等）。
- possible_keys: MySQL 可能使用的索引列表。
- key: MySQL 在执行此操作时实际使用的索引。
- key_len: 使用的索引的长度。
- ref: 显示索引使用的列或常量。
- rows: MySQL 估计从表中读出的行数。
- filtered: 表示表数据的过滤程度。
- Extra: 其他额外的信息。



**select_type**表示查询的类型，它的常用取值有:

1. SIMPLE，表示此查询不包含 UNION 查询或子查询。
2. UNION, 表示此查询是使用UNION语句的第二个或后面的SELECT。
3. SUBQUERY, 子查询中的第一个 SELECT。



**type** 列包括了MySQL在表中查找所需行的方式，是最重要的性能参数之一。下面是各个值的详细含义：

- system: 这是最小范围的类型，仅有一行匹配，这是系统表或某些情况下使用的常量表。这个类型很少看见，一般值为 NULL。
- const: MySQL通过查询常量表，只取回一行数据来优化查询。例如在执行主键查询时，一个const类型的查询会被使用。
- eq_ref: 此类型的连接是索引连接，对于每个索引键，表中只有一条记录与之匹配，常见于主键或唯一索引的查询。
- ref: 和eq_ref类型类似，区别在于它是非唯一索引（非主键索引）查询，返回匹配某个单独的值的所有行。常见于使用索引查找某个特定值的查询。
- range: 此类型的连接方式使用了索引来查找指定范围内的值，不同于ref，它可包含重复的行。
- index: 此类型的查询与全部的表行进行比较，用于优化以索引键值作为限制的查询，效率比表扫描高，但比前面的类型慢。
- all: 此类型的查询与全部的表行进行比较，是最慢的一种查询方式。这通常意味着MySQL无法使用索引，必须进行全表扫描。



### show processlist

返回进程供我们分析



### 其他分析

```sql
--查看每个客户端的连接数
select client_ip,count(client_ip) as client_num 
from (
     selectsubstring_index(host,':' ,1) as client_ip 
     fromprocesslist ) as connect_info 
group by client_ip 
order by client_num desc;

--查看进程时间
select * 
from information_schema.processlist 
where Command != 'Sleep' 
order by Time desc;

--杀死超时进程
select concat('kill ', id, ';')
from information_schema.processlist 
where Command != 'Sleep' and Time > 300
order by Time desc;
```

 

# 面试 SQL

### 累计求和

| id   | money |
| ---- | ----- |
| 1    | 10    |
| 2    | 20    |
| 3    | 30    |
| 4    | 40    |
| 5    | 50    |

表如上，要求结果id1是10，id2是30，id3是60，以此类推。

```sql
-- 两表笛卡尔积，过滤条件主键>=，按求和列分组
SELECT a.id,sum(b.money) 
FROM nm a
JOIN nm b
ON a.id>=b.id
GROUP BY a.id
```



### 获取 TOP-N 

问题：
获取一个表中前5个最高工资的员工名字和对应的工资？

答案：
使用窗口函数`row_number()`配合`order by`实现：

```
SELECT last_name, salary 
FROM (SELECT last_name, salary, row_number() OVER (ORDER BY salary DESC) AS rank_num 
      FROM employees) 
WHERE rank_num <= 5;
```



### 第二大值

如何在不使用子查询或联接的情况下，找出一个表中的第二个最大值？

```
SELECT MAX(salary)
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
```



### 最近的10个数

问题：
查找一个订单表中记录的离给定日期最近的10个日期，假设该表中包含一列名为 `order_date` 的日期列。

答案：
使用`ORDER BY` 排序和 `LIMIT` 子句限制查询结果返回前10行：

```
SELECT order_date 
FROM orders
ORDER BY ABS(DATEDIFF(order_date, '2023-06-01')) 
LIMIT 10;
```

















 





































