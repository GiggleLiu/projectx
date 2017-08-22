Project X
=====================
This is a new repository meant for easy cooperation, with Model-View-Controler architecture.
Previous developments using tests are no longer maintained.

## Introduction to MVC
将应用程序划分为三种组件，模型 - 视图 - 控制器（MVC）设计定义它们之间的相互作用。
* 模型（Model） 用于封装与应用程序的业务逻辑相关的数据以及对数据的处理方法。“ Model ”有对数据直接访问的权力，例如对数据库的访问。“Model”不依赖“View”和“Controller”，也就是说， Model 不关心它会被如何显示或是如何被操作。但是 Model 中数据的变化一般会通过一种刷新机制被公布。为了实现这种机制，那些用于监视此 Model 的 View 必须事先在此 Model 上注册，从而，View 可以了解在数据 Model 上发生的改变。（比较：观察者模式（软件设计模式））
* 视图（View）能够实现数据有目的的显示（理论上，这不是必需的）。在 View 中一般没有程序上的逻辑。为了实现 View 上的刷新功能，View 需要访问它监视的数据模型（Model），因此应该事先在被它监视的数据那里注册。
* 控制器（Controller）起到不同层面间的组织作用，用于控制应用程序的流程。它处理事件并作出响应。“事件”包括用户的行为和数据 Model 上的改变。

In this project, Models are persistent data like state ansatz in models/\*.py, problems.py, observables.py.
Controllers are those functions in controllers.py, where we defined functionalities. View is run.py, this is how we access these functionalities.

## Dependancy
* **qstate** https://159.226.35.226/jgliu/qstate.git
* **PoorNN** https://159.226.35.226/jgliu/PoorNN.git
* **climin** https://159.226.35.226/jgliu/climin.git

* *other packages that can be installed with pip*

## How to use

If you are a developer, you can write your own state ansatz and put it into models/, append new functionalities into controllers.py,
and finally, create your own version of run\_local.py (filename contained in .gitignore).

Also you can write notes under notes/.
