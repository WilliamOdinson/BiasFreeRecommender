## 关于数据集

## 背景

这是一个包含来自Yelp的用户评论和业务数据的数据集。它为研究者和分析师提供了一个平台，通过分析用户评论和业务评价来探索消费者行为和业务趋势。数据集中包含了各种不同类型的商家和来自多个地区的用户反馈。

## 内容

该数据集由`yelp_academic_dataset_review.csv`和`yelp_academic_dataset_business.csv`的CSV文件组成。

### `yelp_academic_dataset_review.csv`包括以下列：

- `user_id`: 用户的唯一标识符
- `review_id`: 评论的唯一标识符
- `text`: 评论内容
- `votes.cool`: 被认为是“酷”的投票数量
- `business_id`: 商家的唯一标识符
- `votes.funny`: 被认为是“有趣”的投票数量
- `stars`: 评分（星级）
- `date`: 评论日期
- `type`: 数据类型
- `votes.useful`: 被认为是“有用”的投票数量

### `yelp_academic_dataset_business.csv`包括以下列：

- `business_id`: 商家在Yelp上的唯一标识符。
- `attributes.Ambience.divey`: 描述商家的环境是否偏向非正式或“潜水吧”风格。
- `attributes.Dietary Restrictions.vegan`: 商家是否提供素食选项。
- `attributes.Happy Hour`: 商家是否提供欢乐时光优惠。
- `hours.Thursday.open`: 周四商家的开门时间。
- `attributes.Order at Counter`: 是否在柜台点餐。
- `attributes.Hair Types Specialized In.africanamerican`: 商家是否专门提供针对非洲裔美国人的发型服务。
- `attributes.Hair Types Specialized In.kids`: 商家是否提供儿童发型服务。
- `attributes.BYOB`: 商家是否允许带酒入内（Bring Your Own Bottle）。
- `hours.Friday.open`: 周五的开门时间。
- `categories`: 商家所属的类别。
- `latitude`: 商家的纬度。
- `attributes.Outdoor Seating`: 商家是否有室外座位。
- `attributes.Alcohol`: 商家提供的酒精饮料类型。
- `attributes.Ambience.classy`: 环境是否优雅。
- `attributes.Payment Types.mastercard`: 是否接受万事达卡付款。
- `attributes.Parking.lot`: 是否有停车场。
- `attributes.Ambience.touristy`: 环境是否具有旅游景点的特色。
- `attributes.Corkage`: 是否允许客人自带酒水并收取开瓶费。
- `hours.Tuesday.open`: 周二的开门时间。
- `attributes.Good For.brunch`: 是否适合早午餐。
- `attributes.Payment Types.amex`: 是否接受美国运通卡。
- `name`: 商家名称。
- `hours.Monday.open`: 周一的开门时间。
- `attributes.Waiter Service`: 是否有服务员服务。
- `attributes.Parking.street`: 是否有街边停车位。
- `attributes.Ambience.hipster`: 环境是否属于潮流或“小众”风格。
- `attributes.BYOB/Corkage`: 是否允许自带酒水并明确是否需要开瓶费。
- `attributes.Hair Types Specialized In.straightperms`: 商家是否提供直发或烫发服务。
- `attributes.Music.live`: 是否提供现场音乐。
- `attributes.Dietary Restrictions.dairy-free`: 是否提供无乳制品选项。
- `attributes.Music.background_music`: 是否有背景音乐。
- `attributes.Price Range`: 价格范围。
- `attributes.Good For.breakfast`: 是否适合作为早餐场所。
- `attributes.Parking.garage`: 是否有车库停车设施。
- `attributes.Music.karaoke`: 是否提供卡拉OK。
- `attributes.Good For Dancing`: 是否适合跳舞。
- `review_count`: 商家的评论数量。
- `attributes.Hair Types Specialized In.asian`: 商家是否专门提供亚洲人发型服务。
- `state`: 商家所在的州。
- `attributes.Accepts Credit Cards`: 是否接受信用卡支付。
- `hours.Friday.close`: 周五的关门时间。
- `attributes.Good For.lunch`: 是否适合午餐。
- `attributes.Good For Kids`: 是否适合儿童。
- `attributes.Parking.valet`: 是否提供代客泊车服务。
- `attributes.Take-out`: 是否提供外卖服务。
- `full_address`: 商家的完整地址。
- `hours.Thursday.close`: 周四的关门时间。
- `attributes.Hair Types Specialized In.coloring`: 商家是否提供染发服务。
- `attributes.Payment Types.cash_only`: 是否只接受现金支付。
- `attributes.Good For.dessert`: 是否适合甜点。

- `attributes.Music.video`: 是否播放音乐视频。
- `attributes.Dietary Restrictions.halal`: 是否提供清真食品。
- `attributes.Takes Reservations`: 是否接受预订。
- `hours.Saturday.open`: 周六的开门时间。
- `attributes.Ages Allowed`: 允许的顾客年龄。
- `attributes.Ambience.trendy`: 环境是否时尚。
- `attributes.Delivery`: 是否提供送货服务。
- `hours.Wednesday.close`: 周三的关门时间。
- `attributes.Wi-Fi`: 是否提供Wi-Fi。
- `open`: 商家是否还在营业。
- `city`: 商家所在城市。
- `attributes.Payment Types.discover`: 是否接受Discover卡支付。
- `attributes.Wheelchair Accessible`: 是否无障碍设施。
- `attributes.Dietary Restrictions.gluten-free`: 是否提供无麸质食品。
- `stars`: 商家的平均评分。
- `attributes.Payment Types.visa`: 是否接受Visa卡支付。
- `type`: 数据类型。
- `attributes.Caters`: 是否提供餐饮服务。
- `attributes.Ambience.intimate`: 环境是否亲密。
- `attributes.Music.playlist`: 是否有播放音乐播放列表。
- `attributes.Good For.latenight`: 是否适合深夜消费。
- `attributes.Good For.dinner`: 是否适合晚餐。
- `attributes.Coat Check`: 是否提供衣帽间服务。
- `longitude`: 商家的经度。
- `hours.Monday.close`: 周一的关门时间。
- `attributes.Hair Types Specialized In.extensions`: 商家是否提供接发服务。
- `hours.Tuesday.close`: 周二的关门时间。
- `hours.Saturday.close`: 周六的关门时间。
- `attributes.Good for Kids`: 是否适合儿童。
- `attributes.Parking.validated`: 是否提供停车验证。
- `hours.Sunday.open`: 周日的开门时间。
- `attributes.Accepts Insurance`: 是否接受保险。
- `attributes.Music.dj`: 是否有DJ音乐。
- `attributes.Dietary Restrictions.soy-free`: 是否提供无大豆产品。
- `attributes.Has TV`: 是否有电视。
- `hours.Sunday.close`: 周日的关门时间。
- `attributes.Ambience.casual`: 环境是否休闲。
- `attributes.By Appointment Only`: 是否只接另通过预约。
- `attributes.Dietary Restrictions.kosher`: 是否提供符合犹太教规的食品。
- `attributes.Dogs Allowed`: 是否允许狗进入。
- `attributes.Drive-Thru`: 是否有得来速服务。
- `attributes.Dietary Restrictions.vegetarian`: 是否提供素食选项。
- `hours.Wednesday.open`: 周三的开门时间。
- `attributes.Noise Level`: 噪音等级。
- `attributes.Smoking`: 是否允许吸烟。
- `attributes.Attire`: 顾客着装要求。
- `attributes.Hair Types Specialized In.curly`: 商家是否专门提供针对卷发的服务。
- `attributes.Good For Groups`: 是否适合团体。
- `neighborhoods`: 商家所在的街区。
- `attributes.Open 24 Hours`: 是否24小时开放。
- `attributes.Ambience.romantic`: 环境是否浪漫。
- `attributes.Hair Types Specialized In.perms`: 商家是否提供烫发服务。
- `attributes.Music.jukebox`: 是否有自动点唱机。
- `attributes.Ambience.upscale`: 环境是否高档。

## 读取文件的代码片段

在Python中，您可以使用pandas库来读取CSV文件，如下所示：

```python
import pandas as pd

yelp_review = pd.read_csv("yelp_academic_dataset_review.csv")
yelp_business = pd.read_csv("yelp_academic_dataset_business.csv")
```
