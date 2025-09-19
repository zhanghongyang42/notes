官网：https://docs.scala-lang.org/zh-cn/tour/tour-of-scala.html

教程：http://twitter.github.io/scala_school/zh_cn/

博客：https://hongjiang.info/scala/



# 简介

Scala 运行在 Java 虚拟机上，并兼容现有的 Java 程序。

Scala中一切皆表达式。Scala是一种**纯面向对象**的语言。Scala是一门**函数式编程**语言。



# 安装

详见 安装



# 关键字

关键字不用理解，关键字是scala语言的原始的指令集合。

| >:      |
| ------- |
| forSome |
| lazy    |
| sealed  |
| yield   |
| <%      |
| :       |
| <:      |
| #       |
| -       |



# 表达式

在Scala中，表达式是指可求值的代码片段。所有的表达式都有一个类型和一个返回值，即使返回值是Unit（表示无返回值）

有些**关键字用于定义表达式**，有些不行，比如class，但是可以用于**组成表达式**。表达式之间又可以互相嵌套组合函数式编程），构成scala语言的基本元素。

基本元素都是表达式的好处就是可以函数式编程。



### 非表达式元素

注意，scala中**关键字不是表达式**，因为它不返回结果，也没有类型。scala中的所有元素都由关键字直接或间接去定义。

注意，scala中**定义类不是表达式**，因为它不返回结果，但是初始化对象是表达式。类不是表达式，但是类中可以包括表达式。

注意，scala中**类型也不是表达式**，因为类型本质上是一个类。

注意，scala中**赋值运算符不是表达式**，因为它不返回结果，也没有类型。



### 表达式元素

除了类、类型、关键字等，还需要一些表达式来帮助scala完成一些基本概念的定义。



对象表达式：类实例化为对象，对象也是表达式。

```scala
new Greeter("Hello, ", "!")	// 实例化
```

方法表达式：方法也是表达式。方法可以定义在类、对象（静态方法）、特质、包对象中

```scala
def square(x:Int):Int = x*x		// 定义

print("Hello, world")			// 调用
```

值表达式：数值表达式、字符串表达式、布尔表达式。值是一个特殊的 实例化的对象

```scala
2 
"Hello, world"
true
```

常量变量表达式：在Scala中，可以使用`val`关键字定义常量，使用`var`关键字定义变量。定义常量或变量时需要指定类型或者赋值

```scala
val x:Int		// 常量
var y:Int		// 变量
var y:Int = 2.5 // 组合表达式
```

函数表达式：函数本质上是一个值，是一个对象。以下函数相关的都是表达式。

```scala
val f:(Int)=>Int		// 函数是一个特殊类型的常量
(x:Int)=>x*x			// 函数字面量是一个特殊的值

val f = (x:Int)=>x*x	// 函数的一般定义
f(2)					// 函数的一搬调用			
```

运算符表达式：运算符本质上是对方法调用的漂亮写法。是特殊的方法表达式

```scala
2 + 3 * 4	// 算术运算符
x > 5		// 关系运算符
a && b		// 逻辑运算符
```

条件表达式：包括if-else语句和match语句

```scala
if (x < y) {
  println("x is less than y")
} else if (x > y) {
  println("x is greater than y")
} else {
  println("x is equal to y")
}
```

循环表达式：循环结构语句：`while`、`for`、`do while`都是表达式

```scala
val result = for (i <- nums) yield i * 2

for (i <- nums) {println(i)}
```

```scala
var i = 0
while (i < 5) {
    println(i)
    i = i + 1
}

var i = 0
do {
    println(i)
    i = i + 1
} while (i < 5)
```

代码块表达式：`{}`包围起来的几个表达式，最后一个表达式的结果是代码块的值

```scala
{
val x = 1 + 1
x + 1
}
```



# 值、常量变量

在Scala中，值有数值、字符串、布尔 等

```scala
2 
"Hello, world"
true
```



在Scala中，可以使用`val`关键字定义常量，使用`var`关键字定义变量。定义常量或变量时需要指定类型或者赋值

```scala
val x:Int		// 常量
var y:Int		// 变量
var y:Int = 2.5 // 组合表达式
```



# 运算符

运算符就是方法（更漂亮）。运算符方法被定义在 使用运算符的数字字符串等 类型类的伴生对象中（伴生对象使运算符方法分离出来，避免对原来类产生影响）



### 算术运算符

```scala
println("a + b = " + (a + b) );		// v1 + v2实际上等价于v1.+(v2)
println("a - b = " + (a - b) );
println("a * b = " + (a * b) );
println("b / a = " + (b / a) );
println("b % a = " + (b % a) );
```



### 关系运算符

```scala
println("a == b = " + (a == b) );
println("a != b = " + (a != b) );
println("a > b = " + (a > b) );
println("a < b = " + (a < b) );
println("b >= a = " + (b >= a) );
println("b <= a = " + (b <= a) );
```



### 逻辑运算符

```scala
println("a && b = " + (a&&b) );
println("a || b = " + (a||b) );
println("!(a && b) = " + !(a && b) );
```



### 位运算符

| 运算符 | 描述           | 实例                                                         |
| :----- | :------------- | :----------------------------------------------------------- |
| &      | 按位与运算符   | (a & b) 输出结果 12 ，二进制解释： 0000 1100                 |
| \|     | 按位或运算符   | (a \| b) 输出结果 61 ，二进制解释： 0011 1101                |
| ^      | 按位异或运算符 | (a ^ b) 输出结果 49 ，二进制解释： 0011 0001                 |
| ~      | 按位取反运算符 | (~a ) 输出结果 -61 ，二进制解释： 1100 0011， 在一个有符号二进制数的补码形式。 |
| <<     | 左移动运算符   | a << 2 输出结果 240 ，二进制解释： 1111 0000                 |
| >>     | 右移动运算符   | a >> 2 输出结果 15 ，二进制解释： 0000 1111                  |
| >>>    | 无符号右移     | A >>>2 输出结果 15, 二进制解释: 0000 1111                    |



### 赋值运算符

```scala
c = a + b;
c += a ;
c -= a ;
c *= a ;
c /= a ;
c %= a ;
c <<= 2 ;
c >>= 2 ;
c >>= a ;
c &= a ;
c ^= a ;
c |= a ;
```



### 优先级

| 类别 |               运算符               | 关联性 |
| :--: | :--------------------------------: | :----: |
|  1   |               () []                | 左到右 |
|  2   |                ! ~                 | 右到左 |
|  3   |               * / %                | 左到右 |
|  4   |                + -                 | 左到右 |
|  5   |             >> >>> <<              | 左到右 |
|  6   |             > >= < <=              | 左到右 |
|  7   |               == !=                | 左到右 |
|  8   |                 &                  | 左到右 |
|  9   |                 ^                  | 左到右 |
|  10  |                 \|                 | 左到右 |
|  11  |                 &&                 | 左到右 |
|  12  |                \|\|                | 左到右 |
|  13  | = += -= *= /= %= >>= <<= &= ^= \|= | 右到左 |
|  14  |                 ,                  | 左到右 |



# 流程控制

### 条件表达式

```scala
if (x < y) {
  println("x is less than y")
} else if (x > y) {
  println("x is greater than y")
} else {
  println("x is equal to y")
}
```



### 模式匹配

模式匹配是检查某个值（value）是否匹配某一个模式的机制。是Java中的`switch`语句的升级版，同样可以用于替代一系列的 if/else 语句。

```scala
def matchTest(x: Int): String = x match {
  case 1 => "one"
  case 2 => "two"
  case _ => "other"
}
```



### 循环表达式

循环结构语句：`while`、`for`、`do while`

```scala
val result = for (i <- nums) yield i * 2

for (i <- nums) {println(i)}
```

```scala
var i = 0
while (i < 5) {
    println(i)
    i = i + 1
}
```

```scala
var i = 0
do {
    println(i)
    i = i + 1
} while (i < 5)
```



# 函数

函数是带有参数的表达式，函数本质上是一个值，是一个对象



### 函数

```scala
// 定义
val addOne = (x:Int)=>x+1

// 调用
addOne(2)
```



### 匿名函数

一般用于函数作为某个方法参数，就直接传入匿名函数即可

```scala
// 一般写法
(x:Int)=> x+1

// 简化写法
_+1
```



### 高阶函数

参数或者返回值 是 函数或者方法 的函数

```scala
// 举例map函数,参数是（匿名）函数
val salaries = Seq(20000, 70000, 40000)
val newSalaries = salaries.map(_ * 2)

//返回值是函数
def urlBuilder(endpoint: String, query: String): (String, String) => String = {
  (endpoint: String, query: String) => endpoint + String
}
```



# class

Scala中的类是用于创建对象的蓝图，其中包含了方法、常量、变量、类（内部类）、特质、对象，这些统称为成员。

```scala
class Person(val name: String, var age: Int) {
  val salaries = Seq(20000, 70000, 40000)
  def greet(): Unit = {println(s"Hello, my name is $name and I am $age years old.")}
}
```

Scala 会以Main单例对象，main方法作为程序入口。不用管为什么。

```scala
object Main {
  def main(args: Array[String]): Unit = {
    val person = new Person("John", 30)
    person.greet() 
  }
}
```



### 构造方法

构造方法在类实例化的时候被自动调用，进行对象的初始化。

Scala中分为主构造方法和辅助构造方法。

主构造方法是类定义的一部分，在类名后面直接定义。

辅助构造方法可以定义多个。使用`this()`作为辅助构造方法的标志；每个辅助构造器，都必须以其他辅助构造器，或者主构造器的调用作为第一句



以下是一个使用主构造方法和辅助构造方法的Scala类示例：

```scala
class Person(val name: String, var age: Int) {	//主构造方法,主构造方法中带有val和var的参数是公有的，否则仅仅在类中可见
  def this() = this("unknown", 0)				//辅助构造方法
  def this(name: String) = this(name, 0)		//辅助构造方法
  def this(age: Int) = this("unknown", age)		//辅助构造方法

  def greet(): Unit = {println(s"Hello, my name is $name and I am $age years old.")}
}

object Main {
  def main(args: Array[String]): Unit = {
    val person1 = new Person("John", 30)
    person1.greet() // 输出 "Hello, my name is John and I am 30 years old."

    val person2 = new Person
    person2.greet() // 输出 "Hello, my name is unknown and I am 0 years old."

    val person3 = new Person("Mike")
    person3.greet() // 输出 "Hello, my name is Mike and I am 0 years old."

    val person4 = new Person(40)
    println(person4.age)	// 主构造方法中带有val和var的参数是公有的，否则仅仅在类中可见
  }
}
```



### 方法

类的成员，和函数相似，方法中可以嵌套方法。

```scala
def getSquareString(input: Double): String = {
    val square = input * input
    square.toString
}
```



##### 柯里化

柯里化允许一个接收多个参数的方法转化为接收一系列单个参数的方法。



举个例子，假设有一个接收两个整数参数的函数：

```scala
def add(x: Int, y: Int) = x + y
```

可以将其转化为一个柯里化函数，如下所示：

```scala
def add(x: Int)(y: Int) = x + y
```

这样，调用该函数时就必须用多次调用来依次提供每个参数。例如：

```scala
val result = add(3)(5)
```



柯里化函数调用第一次参数返回的应该是一个匿名函数

```scala
def add(x:Int) = (y:Int) => x + y
```



##### 传名参数

传值参数，是直接传入参数值，方法内多次使用时避免计算

传名参数，是直接将参数表达式传入，方法内不使用不计算。

```scala
// 传名参数用 :=> ，普通参数用 :
def calculate(input :=> Int) = input * 37
```



`Java中只有方法，C中只有函数，而C++里取决于是否在类中。`



### 内部类

```scala
//定义一个内部类
class Graph {
    class Node {def connectTo(node: Node) {}}
    var nodes: List[Node] = Nil
}

//初始化内部类实例，注意内部类类型由外部类实例决定 graph1.Node
val graph1: Graph = new Graph
val node1: graph1.Node = graph1.newNode

//内部类中用到的内部类类型变量，实例化后，受外部类实例的影响
val node2: graph1.Node = graph1.newNode
node1.connectTo(node2)//没问题
val graph2: Graph = new Graph
val node3: graph2.Node = graph2.newNode
node1.connectTo(node3)//java没问题，scala会报错

//除非定义的时候用 外部类#内部类 的形式
class Graph {
    class Node {def connectTo(node:Graph#Node) {}}
    var nodes: List[Node] = Nil
}
```



### 访问修饰符

在Scala中，可以使用以下四种访问修饰符来限制成员的可见范围：

1. `private`: 只能在包含该成员的类或对象内部直接访问，包括其伴生对象。

2. `protected`: 只能在包含该成员的类或对象内部及其子类中直接访问，包括其伴生对象。

3. `private[package]`: 只能在指定的包及其子包内直接访问，包括其伴生对象。

4. 没有修饰符（默认访问级别）：可以被任意包中的任意类或对象访问，包括其伴生对象。

以下是一个使用各种访问修饰符的示例：

```scala
private val id: Int = 100
private[Cat] val distance = 100
```



### 类的继承

```scala
class Animal {
  def sound(): String = "undefined"
}

class Cat extends Animal {
  override def sound(): String = "meow"
}
```



### 类的多态

```scala
abstract class Vehicle {
  def accelerate(): Unit
}

class Car extends Vehicle {
  override def accelerate(): Unit = println("This car is accelerating!")
}

class Bicycle extends Vehicle {
  override def accelerate(): Unit = println("This bicycle is accelerating!")
}

object Demo {
  def main(args: Array[String]): Unit = {
    val vehicle1: Vehicle = new Car()
    val vehicle2: Vehicle = new Bicycle()
    
    vehicle1.accelerate()
    vehicle2.accelerate()
  }
}
```



# 类层次

scala中类的层次结构如下：

![Scala Type Hierarchy](https://raw.githubusercontent.com/zhanghongyang42/images/main/unified-types-diagram.svg)

Any 是所有类型的超类型，也称为顶级类型。它定义了一些通用的方法如`equals`、`hashCode`和`toString`。

AnyVal 基本类型：Byte、Short、Int、Long、Float、Double、Char  、Boolean、Unit。

AnyRef 引用类型，相当于java的Object。



类型转换

![Scala Type Hierarchy](https://raw.githubusercontent.com/zhanghongyang42/images/main/type-casting-diagram.svg)

Nothing和Null

没有一个值是`Nothing`类型的。它的用途之一是给出非正常终止的信号，如抛出异常、程序退出或者一个无限循环。

`Null`是所有引用类型的子类型。它有一个单例值由关键字`null`所定义。`Null`主要是使得Scala满足和其他JVM语言的互操作性。

Unit 类型也只有一个值()。



复合类型：可以给一个变量多个类型，用with

```scala
def cloneAndReset(obj: Cloneable with Resetable): Cloneable = {}
```



# 集合

scala.collection包下有许多集合，下面介绍常用的几种。



可变集合`scala.collection.mutable` 需要导入包

不可变集合 `scala.collection.immutable` 默认已经导入



可变集合场景：对同一个数据集合进行多次修改和更新。多线程环境下考虑数据的线程安全问题。

不可变集合场景：多线程环境下或者需要数据不变（只读）场景。可以保证线程安全和数据不变性



### 操作符

可变集合增删改会改变集合，不可变集合增删改会产生新集合。



**通用操作符**

:: 连接元素和集合，全部都是元素时，末尾加空集合Nil

```scala
val list = 1 :: 2 :: 3 :: Nil
```

::: 连接两个集合

```scala
val newList = list1 ::: list2	// 输出: List(1, 2, 3, 4, 5, 6)
```

++  连接多个集合

```scala
val newList = list1 ++ list2	// 输出: List(1, 2, 3, 4, 5, 6)
```

+：将一个元素添加到列表的头部

```scala
val newList = 1 +: list	// 输出: List(1, 2, 3)
```

:+ 将一个元素添加到列表的尾部

```scala
val newList = list :+ 4	// 输出: List(1, 2, 3, 4)
```



**可变集合操作符**

++= 将一个集合加入另一个集合尾部

```scala
list1 ++= list2	// list1：ListBuffer(1, 2, 3, 4, 5, 6, 7, 8)
```

+= 将一个元素添加到列表的尾部

```scala
listBuffer += 3	// 输出: ListBuffer(1, 2, 3)
```

-=    --=  删除元素或集合



### Array 

有序，元素类型一致



Array 长度不可变

```scala
// 定义
val arrayTest = new Array[Int](10)
val arrayTest = Array(1,2,3)
```

ArrayBuffer 长度可变

```scala
// 定义
val arrayTest = new ArrayBuffer[Int]()

// 增
arrayTest+=("Mysql","Html")
arrayTest++=Array("Hadoop","Hive","Sqoop")

// 删
arrayTest-= "Mysql"
arrayTest--= Array("JavaSE","Sqoop")

// 改查
arrayTest(0)="CSS"

// 遍历
for(item <- arrayTest) {println(item)}

// 函数操作
array6.sum
array6.max
array6.sorted
```

多维数组

```scala
val dim = Array.ofDim[Double](3,4)
dim(1)(1) = 11.11
println(dim.mkString(","))
```



### List

有序，元素类型不一致



List 长度和数据不可变

```scala
// 定义
val list1 = List("hello",20,5.0f)
val list1 = 1 :: 2 :: 3 :: Nil

// 增
val newList = 0 :: list // 在头部添加元素
val anotherList = list :+ 4 // 在尾部添加元素
val anotherList = List(1, 2, 3)++List(4, 5, 6) //集合合并

// 查
anotherList(0)
anotherList.range(2, 6)
list1.head
list1.last
list1.tail
list1.tail.head
list1.tail.tail.head
list1.isEmpty
list1.foreach((x: String) => println(x))

// 函数
val newList = list.reverse
val newList = list.sorted
val newList = list.distinct
val newList = list.map(_ * 2)
val anotherList = list.filter(_ % 2 == 0)
val lenSum = list.reduce((x: Int, y: Int) => x + y)
mapValues
groupBy
...
```

ListBuffer 长度和数据可变

```scala
// 定义
val lb= ListBuffer()
val lb = ListBuffer.empty[Int]

// 增
lb += 1
lb += 2
lb += 3
lb ++= List("javaSE","JavaEE")

// 删
lb -= 3
lb --= List("JavaEE")

// 改
lb(1) = 5
lb.insert(1, 4)
```



### Set

元素不重复集合，可变set和不可变set类名相同。

```scala
// 定义
import scala.collection.mutable.Set
val set2 = Set(1, 2, 3)

// 增删
set2 += 5
set3 -= 1

// 集合运算
val unionSet = set ++ set2	// 并集	
val jiaojiSet=set & set2	// 交集	
val chajiSet=set &~ set2	// 差集	
```



### Tuple

元组，不同类型元素，不可变的集合。

元组类从Tuple2到Tuple22 共22个，也就是说元组最多22个元素。

```scala
// 定义
val ingredient = ("Sugar" , 25)

// 查
println(ingredient._1)
```



### Map

键值对的集合，键是唯一的，键的类型是一致的。

```scala
// 不可变Map
val myMap = Map("apple" -> 1, "banana" -> 2, "orange" -> 3)	// ->是方法 返回值是tuple2

// 查询
map2.get("address")
map2.keys
map2.values

// 遍历
for(item<-map){
    println(item._1)
    println(item._2)
}
for((key,value) <- map){
    println(key)
    println(value)
}



// 可变Map
import scala.collection.mutable.Map
val myMutableMap = Map[String, Int]()

// 添加
myMutableMap += ("apple" -> 1)

// 更新
myMutableMap("apple") = 4

// 删除
map2.-("address")
```



# trait

特质 (Traits) 用于在类之间或对象 (Objects)之间共享程序接口和字段。类似于Java 8的接口。

特质不能被实例化，没有参数。

scala中是单继承，多混入，特质可以被混入



继承

```scala
//特质定义的时候，抽象方法不实现方法，泛型不指定类型，很常用
trait Iterator[A] {
  def hasNext: Boolean
  def next(): A
}

class IntIterator(to: Int) extends Iterator[Int] {
  override def hasNext: Boolean = 0 < to
  override def next(): Int = to + 5
}
```



混入

一个类只能有一个父类但是可以有多个混入。

```scala
class B {
  val message = "I'm an instance of class B"
}
trait C {
  def loudMessage = message.toUpperCase()
}

class D extends B with C
```



自类型

```scala
trait User {
    def username: String
}

trait Tweeter {
    //this: Type => ，继承或者混入该特质的时候，指定的Type必须一起混入
    this: User =>  
    def tweet(tweetText: String) = println(s"$username: $tweetText")
}

class VerifiedTweeter(val username_ : String) extends Tweeter with User {
	def username = s"real $username_"
}
```



# object

1. 定义静态成员变量和方法。在 `object` 中定义的成员变量和方法都是静态的，可以通过 `object` 名称直接访问。
2. 定义实现单例模式的类。当 `object` 没有特质或父类时，会被看作是一个单例对象，只有一个实例存在。

```scala
object MyObject {
  val x: Int = 1
  def add(a: Int, b: Int): Int = a + b
}

val newId: Int = IdFactory.add(1,2)
```



### 伴生对象

当一个单例对象和某个类共享一个名称时，这个单例对象称为 *伴生对象*。这个类称为伴生类。类和它的伴生对象可以互相访问其私有成员。

`类和它的伴生对象必须定义在同一个源文件里`



提供静态的成员和方法，实现通用功能

```scala
class Person(val name: String, val age: Int)

object Person {
  def staticMethod(): Unit = println("this is a static method") 
}

// 调用伴生对象中的静态方法
Person.staticMethod()
```



### apply

在 Scala 中，apply 和 unapply 是特殊的方法，它们在编译期间被编译器特殊处理，因此在调用时有些不同于普通方法。

apply 方法调用，可以省略掉方法名，直接使用类名加上参数列表的形式调用。一般是接收参数返回对象。

```scala
class Person(val name: String, val age: Int)

object Person {
  def apply(name: String, age: Int): Person = new Person(name, age)
}

// 伴生对象的 apply 方法
val p = Person("Alice", 25)
```



### 提取器

提取器（Extractor）是一个 object 对象，定义了一个或多个 unapply 或 unapplySeq 方法。

提取器通常用于模式匹配中，利用 unapply 或 unapplySeq 方法来将某个对象（如集合、类实例等）解构成若干个部分。

```scala
class Person(val name: String, val age: Int)

// 提取器
object Person {
  def unapply(p: Person): Option[(String, Int)] = (p.name, p.age)
}

val alice = new Person("Alice", 25)

// 直接调用unapply方法
val (name, age) = Person.unapply(alice)
val name, age = Person.unapply(alice)

val Person(name, age) = alice //实际上也是得到了(name, age)或者name, age

// 模式匹配调用unapply方法，可以省略unapply
alice match {
  // 这里case Person(name, age) 其实就是case (name, age)
  case Person(name, age) => println(s"$name is $age years old")
}
```



# case class

case class 自动为每个类生成一个伴生对象

1. 伴生对象的 apply 方法：case class 伴生对象的 apply 方法，用于创建新的实例。

2. 伴生对象的 unapply 方法：case class 伴生对象的 unapply 方法，用于将类实例解构为其成员属性，并返回一个 Option 类型的元组。

3. toString 方法：case class 自动重载了 toString 方法，输出类实例的字符串表示，便于调试和输出。

4. equals 方法：case class 自动重载了 equals 方法，用于比较类实例是否相等。在比较两个类实例时，只有所有成员属性都相等时，它们才被认为是相等的。

5. hashCode 方法：case class 自动重载了 hashCode 方法，用于生成实例的哈希码。

6. copy 方法：case class 自动提供了 copy 方法，可以在不改变原始实例的前提下复制实例，并修改实例的某些属性。

同时，case class 还自动继承了 Product 和 Serializable 等特质，使其能够更方便地进行序列化和反序列化等操作。



```scala
// 定义
case class Person(name: String, age: Int)

// 实例化，调用伴生对象的apply方法
val p = Person.apply("Alice", 20)

// 模式匹配
p match {
    case Person(_, age) if age < 18 => println("underage")
    case Person(_, age) if age >= 18 && age <= 25 => println("young adult")
    case Person(_, age) if age > 25 => println("adult")
    case _ => println("Unknown")
}
```

```scala
//比较时是值比较，不是地址比较
case class Point(x: Int, y: Int)

val p1 = Point(1, 2)
val p2 = Point(1, 2)
val p3 = Point(3, 4)

println(p1 == p2) // true
println(p1 == p3) // false
```



# 模式匹配

模式匹配是检查某个值（value）是否匹配某一个模式的机制。是Java中的`switch`语句的升级版，同样可以用于替代一系列的 if/else 语句。

```scala
def matchTest(x: Int): String = x match {
  case 1 => "one"
  case 2 => "two"
  case _ => "other"
}

//模式守卫
def matchTest(x: Int): String = x match {
  case 1 if true => "one"
  case 2 if true => "two"
  case _ => "other"
}
```

case class 模式匹配

```scala
abstract class Notification
case class Email(sender: String, title: String, body: String) extends Notification
case class SMS(caller: String, message: String) extends Notification
case class VoiceRecording(contactName: String, link: String) extends Notification

//仅匹配类型
def showNotification(notification: Notification): String = {
    notification match {
        case e:Email => s"You got an email from $sender with title: $title"
        case s:SMS => s"You got an SMS from $number! Message: $message"
        case v:VoiceRecording => s"you received a Voice Recording from $name! Click the link to hear it: $link"
  }
}

//匹配类型和参数，使用伴生对象的unapply实现
def showNotification(notification: Notification): String = {
    notification match {
        case Email(sender, title, _) =>s"You got an email from $sender with title: $title"
        case SMS(number, message) =>s"You got an SMS from $number! Message: $message"
        case VoiceRecording(name, link) =>s"you received a Voice Recording from $name! Click the link to hear it: $link"
  }
}
```



# 隐式转换

隐式转换使用 implicit 定义，可以修饰 def，class，val，var。

隐式转换的作用域在当前作用域（大括号）内。



### implicit def

隐式转换方法由编译器自己使用，当前作用域某些代码不通的时候，自动调用隐式转换方法

```scala
// 实现 隐式类型转换 示例
object ImplicitExample {
    // 定义一个隐式方法
    implicit def StringToInt(str: String): Int = Integer.parseInt(str)
  
    // 编译类型不通过,自动调用当前作用域的隐式转换方法
    val i: Int = "123"
}
```



### implicit class

`implicit class` 只能在 `object`、`class` 或 `trait` 中定义。

当某些代码因为类型问题不通的时候，自动转换为隐式转换域的 `implicit class`的类型再执行。

目的是增强某些类型，为这些类型添加新的功能或者方法。

```scala
// 被增强的类
case class Rectangle(width: Double, height: Double)

// 隐式转换类，用于增加作用域内的类。
implicit class RectangleOps(r: Rectangle) {
  def area: Double = r.width * r.height
}

object Main extends App {
  val rect = Rectangle(3.5, 2.0)
  val area = rect.area	// 隐式转换,增强了Rectangle，为其增加了area方法
  println(s"The area of a rectangle with width ${rect.width} and height ${rect.height} is $area.")
}
```



### 隐式参数

隐式参数是方法参数

```scala
// 定义隐式参数值
implicit val greeting: String = "Hello"

// 定义方法中的隐式参数
def greet(name: String)(implicit greeting: String) = println(s"$greeting, $name!")

// 调用方法，不用传入隐式参数值，自动从上下文的隐式参数值中推出。
greet("John")	// 输出 "Hello, John!"
```



隐式参数是类构造方法参数，实现了依赖注入。

可以将业务逻辑与依赖关系解耦，并且可以方便进行单元测试和维护。

```scala
trait CalculatorService {
  def calculate(a: Int, b: Int): Int
}

class AdditionService extends CalculatorService {
  override def calculate(a: Int, b: Int): Int = a + b
}

class SubtractionService extends CalculatorService {
  override def calculate(a: Int, b: Int): Int = a - b
}

// 接收隐式参数实例的类
class Calculator(implicit calculatorService: CalculatorService) {
  def calculate(a: Int, b: Int): Int = calculatorService.calculate(a, b)
}

object Main extends App {
  // 定义隐式参数实例
  implicit val additionService = new AdditionService
    
  // 依赖注入，自动注入隐式参数实例
  val calculator = new Calculator	
  val result = calculator.calculate(3, 2)
  println(s"The result is $result.")
}
```



### 隐式导入

如果想使用其他作用域的隐式转换，需要导入到当前作用域。

```scala
class AminalType
class SwingType{ def  wantLearned(sw : String) = println("兔子已经学会了"+sw)}

object swimming{ implicit def learningType(s : AminalType) = new SwingType}

object test{
    import com.mobin.scala.Scalaimplicit.swimming._		//隐式导入
    val rabbit = new AminalType
    
    // 隐式转换，rabbit是AminalType类型，没有wantLearned方法。使用learningType，隐式转换成SwingType。就可以调用wantLearned方法了
    rabbit.wantLearned("breaststroke")        
}
```



# 泛型

类的泛型用法同java。待整理中。



scala 的泛型可以是协变，逆变，不变的。使用型变，可以让我们直接把**泛型**的继承关系变成**泛型所修饰类**的继承关系。

```scala
//定义有继承关系的泛型，是准备工作
abstract class Animal {def name: String}
case class Cat(name: String) extends Animal
```



协变：协变就是泛型所修饰类的 父类可以接收子类。

协变List[+A]就代表两个类Printer[A]和Printer[A的子类]

```scala
//使用已有的协变类List[+A]演示协变用法
//演示开始
object CovarianceTest extends App {
    //printAnimalNames 方法只接收 List[Animal]
    def printAnimalNames(animals: List[Animal]): Unit = animals.foreach { animal =>println(animal.name)}
    //得到List[Cat]类型实例
    val cats: List[Cat] = List(Cat("Whiskers"), Cat("Tom"))
    //因为List[+A]是协变的，所以可以把 Cat 与 Animal的关系变为 List[Animal]和 List[Cat]的关系，使父类可以接收子类
    printAnimalNames(cats)
}
```



逆变：逆变就是泛型所修饰类 子类可以接收父类。

逆变Printer[-A]就代表两个类Printer[A]和Printer[A的父类]

```scala
//定义一个逆变类Printer[-A]演示逆变用法
abstract class Printer[-A] {def print(value: A): Unit}
class AnimalPrinter extends Printer[Animal] { def print(animal: Animal): Unit = println("The animal's name is: " + animal.name)}
class CatPrinter extends Printer[Cat] { def print(cat: Cat): Unit = println("The cat's name is: " + cat.name)}
//演示开始
object ContravarianceTest extends App {
    //printMyCat 方法只接收 Printer[Cat]
    def printMyCat(printer: Printer[Cat]): Unit = printer.print(Cat("Boots"))
    //得到Printer[Animal]类型实例
    val animalPrinter: Printer[Animal] = new AnimalPrinter
    //因为Printer[-A]是逆变的，所以可以把 Cat 与 Animal的关系变为 Printer[Animal]和 Printer[Cat]的关系，使子类可以接收父类
    printMyCat(animalPrinter)
}
```



不变：不能把泛型的继承关系变成泛型所修饰类的继承关系

不变Printer[A]就代表Printer[A]



泛型上界

```scala
//泛型P在使用时之能是Pet类的子类
abstract class Pet
class PetContainer[P <: Pet](p: P) {def pet: P = p}
```

泛型下界：https://docs.scala-lang.org/zh-cn/tour/lower-type-bounds.html



抽象类型：泛型变量

特质和抽象类可以包含抽象类型成员

```scala
trait Buffer {
  type T
  val element: T
}
```

https://docs.scala-lang.org/zh-cn/tour/abstract-type-members.html



# 包

```scala
// 定义在哪个包,一般与目录相同
package com.runoob 	

// scala 导入包不在顶部，就有作用范围。
import java.awt._ 

//如果存在命名冲突，你需要从项目的根目录导入，在包名称前加上 _root_
import _root_.users._
```



包对象

Scala 提供包对象作为在整个包中方便的共享使用的容器。

包对象中可以定义任何内容，而不仅仅是变量和方法。 例如，包对象经常用于保存包级作用域的类型别名和隐式转换。 包对象甚至可以继承 Scala 的类和特质。

每个包都允许有一个包对象。 在包对象中的任何定义都被认为是包自身的成员。包对象的代码通常放在名为 `package.scala` 的源文件中。

```scala
//定义包对象，将 planted 和 showFruit 放入包 gardening中。
package gardening

package object fruits {
  val planted = List(Apple, Plum, Banana)
    
  def showFruit(fruit: Fruit): Unit = {
    println(s"${fruit.name}s are ${fruit.color}")
  }
}
```



# 面试题

### 懒加载

使用Lazy关键字进行懒加载操作。在一些情况中我们经常希望某些变量的初始化要延迟，并且表达式不会被重复计算。如：

```text
为了缩短模块启动时间，可以将当前不需要的某些工作推迟执行。
保证对象中其他字段的初始化能优先执行。
```



### 闭包

 一个函数把外部的那些不属于自己的对象也包含(闭合)进来。 通俗的来说就是局部变量当全局变量来使用！！！

高阶函数就是闭包，不用实际理解闭包这个概念。

```scala
def minusxy(x: Int) = (y: Int) => x - y

val f1 = minusxy(10)

val f2 = minusxy(10)
```

此处f1,f2这两个函数就叫闭包。



### Option

在Scala语言中，Option类型是一个特殊的类型，它是代表有值和无值的体现，内部有两个对象，一个是Some一个是None，Some代表有返回值，内部有值，而None恰恰相反，表示无值，比如，我们使用Map集合进行取值操作的时候，当我们通过get取值，返回的类型就是Option类型，而不是具体的值。



### 偏函数

偏函数表示用{}包含用case进行类型匹配的操作，这种操作一般用于匹配唯一的属性值，在Spark中的算子内经常会遇到。

```text
val rdd = sc.textFile(路径)
rdd.map{
    case (参数)=>{返回结果}
}
```



### Unit

Scala中的Unit类型类似于java中的void，无返回值。主要的不同是在Scala中可以有一个Unit类型值，也就是（），然而java中是没有void类型的值的。除了这一点，Unit和void是等效的。一般来说每一个返回void的java方法对应一个返回Unit的Scala方法。



### to和until

例如1to10，它会返回Range（1,2,3,4,5,6,7,8,9,10），而1until 10 ，它会返回Range（1,2,3,4,5,6,7,8,9）

也就是说to包头包尾，而until 包头不包尾！



### Nil, Null, None, Nothing

Null是一个trait（特质），是所有引用类型AnyRef的子类型，null是Null唯一的实例。

Nothing也是一个trait（特质），是所有类型Any（包括值类型和引用类型）的子类型，它不在有子类型，它也没有实例，实际上为了一个方法抛出异常，通常会设置一个默认返回类型。

Nil代表一个List空类型，等同List[Nothing]

None是Option 的空标识















































