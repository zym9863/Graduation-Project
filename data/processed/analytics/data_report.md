# 数据集统计报告

本报告由 `uv run python main.py dataset-report` 自动生成。

## 数据规模

| 指标 | 数值 |
| --- | ---: |
| unique_news | 65238 |
| categories | 18 |
| subcategories | 270 |
| train_behaviors | 156965 |
| dev_behaviors | 73152 |
| train_unique_users | 50000 |
| dev_unique_users | 50000 |
| train_impressions | 5843444 |
| dev_impressions | 2740998 |
| train_clicks | 236344 |
| dev_clicks | 111383 |
| train_ctr | 0.0404 |
| dev_ctr | 0.0406 |

## 行为分布摘要

| Split | avg_history_len | p50_history_len | p90_history_len | avg_impression_len | p50_impression_len | p90_impression_len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 32.54 | 19.00 | 78.00 | 37.23 | 24.00 | 91.00 |
| dev | 32.30 | 19.00 | 78.00 | 37.47 | 23.00 | 92.00 |

## 图表清单

- news_category: [news_category_distribution.png](figures/news_category_distribution.png)
- news_subcategory_top: [news_subcategory_top.png](figures/news_subcategory_top.png)
- history_length_train: [history_length_train.png](figures/history_length_train.png)
- history_length_dev: [history_length_dev.png](figures/history_length_dev.png)
- impression_length_train: [impression_length_train.png](figures/impression_length_train.png)
- impression_length_dev: [impression_length_dev.png](figures/impression_length_dev.png)
- hourly_behavior_train: [hourly_behavior_train.png](figures/hourly_behavior_train.png)
- hourly_behavior_dev: [hourly_behavior_dev.png](figures/hourly_behavior_dev.png)
- category_ctr_train: [category_ctr_train.png](figures/category_ctr_train.png)
- category_ctr_dev: [category_ctr_dev.png](figures/category_ctr_dev.png)

## 统计表清单

- news_category: [news_category_distribution.csv](tables/news_category_distribution.csv)
- news_subcategory_top: [news_subcategory_top.csv](tables/news_subcategory_top.csv)
- news_subcategory: [news_subcategory_distribution.csv](tables/news_subcategory_distribution.csv)
- history_length_train: [history_length_train.csv](tables/history_length_train.csv)
- history_length_dev: [history_length_dev.csv](tables/history_length_dev.csv)
- impression_length_train: [impression_length_train.csv](tables/impression_length_train.csv)
- impression_length_dev: [impression_length_dev.csv](tables/impression_length_dev.csv)
- hourly_behavior_train: [hourly_behavior_train.csv](tables/hourly_behavior_train.csv)
- hourly_behavior_dev: [hourly_behavior_dev.csv](tables/hourly_behavior_dev.csv)
- category_ctr_train: [category_ctr_train.csv](tables/category_ctr_train.csv)
- category_ctr_dev: [category_ctr_dev.csv](tables/category_ctr_dev.csv)

## 新闻样例

| news_id | category | subcategory | title | abstract |
| --- | --- | --- | --- | --- |
| N1 | video | news | Lindsey Graham to Trump: "I will hold you accountable" | Senator Lindsey Graham, one of the leading critics of President Trump's drawdown of U.S. troops from Syria, stood alongside senators from both parties Thursday and introduced a ... |
| N10 | health | weight-loss | 7 Running Workouts for Weight Loss | It's all about the fartleks. |
| N100 | sports | baseball_mlb | Astros finally get timely hitting to get back into World Series | WASHINGTON   Somewhere over the eastern skies, the Astros assembled again as one. On an airplane to the nation's capital, the team conferred about their conundrum. A players-onl... |

## 行为样例（Train）

| impression_id | user_id | time | history_len | impression_len | clicked_preview | history_preview |
| --- | --- | --- | ---: | ---: | --- | --- |
| 1 | U13740 | 11/11/2019 9:05:58 AM | 9 | 2 | N55689 | N55189 N42782 N34694 N45794 N18445 N63302 N10414 N19347 |
| 2 | U91836 | 11/12/2019 6:11:30 PM | 82 | 11 | N17059 | N31739 N6072 N63045 N23979 N35656 N43353 N8129 N1569 |
| 3 | U73700 | 11/14/2019 7:01:48 AM | 16 | 36 | N23814 | N10732 N25792 N7563 N21087 N41087 N5445 N60384 N46616 |

## 行为样例（Dev）

| impression_id | user_id | time | history_len | impression_len | clicked_preview | history_preview |
| --- | --- | --- | ---: | ---: | --- | --- |
| 1 | U80234 | 11/15/2019 12:37:50 PM | 15 | 22 | N31958 | N55189 N46039 N51741 N53234 N11276 N264 N40716 N28088 |
| 2 | U60458 | 11/15/2019 7:11:50 AM | 13 | 7 | N23513 | N58715 N32109 N51180 N33438 N54827 N28488 N61186 N34775 |
| 3 | U44190 | 11/15/2019 9:55:12 AM | 9 | 23 | N5940 | N56253 N1150 N55189 N16233 N61704 N51706 N53033 N15634 |
