原始pkl加载之后是列表，列表中的元素是一个trace，
{'timestamp': [1586745795.148996, 1586745795.151688], 'endtime': [1586745796.378302, 1586745795.16016], 'latency': [1.229306, 0.008472], 'http_status': ['200', '200'], 'trace_id': '1027a7c7b2f510909df6375ed2efe952', 'label': 0, 's_t': [('ui-dashboard', 'auth'), ('auth', 'verification-code')]}

所有的pkl都要先经过processing py。processing中转化成df，再dump存储。
时间列精度太高，转化为毫秒 df['endtime'] = pd.to_datetime(df['endtime'] / 1000, unit='ms')
处理完之后，dump dataframe长什么样？

                           trace_id     timestamp  latency  http_status  \
0  df6f230fa9712a90799d39c1c45a5777  1.570798e+15    605.0        304.0   
1  df6f230fa9712a90799d39c1c45a5777  1.570798e+15   1989.0        304.0   
2  cc8da682b10a53aa04c9071ea3fa8ed3  1.570798e+15  14632.0        200.0   
3  cc8da682b10a53aa04c9071ea3fa8ed3  1.570798e+15   8525.0        200.0   
4  cc8da682b10a53aa04c9071ea3fa8ed3  1.570798e+15  10011.0        200.0   

    cpu_use  mem_use_percent  mem_use_amount  file_write_rate  file_read_rate  \
0  0.378031         0.020359    1.420657e+08              0.0             0.0   
1  0.378031         0.020359    1.420657e+08              0.0             0.0   
2  2.406508         0.446083    1.075196e+09              0.0             0.0   
3  8.861334         0.421246    1.027621e+09              0.0             0.0   
4  8.861334         0.421246    1.027621e+09              0.0             0.0   

   net_send_rate  net_receive_rate       endtime  \
0   16600.323566      31423.274824  1.570798e+15   
1   16600.323566      31423.274824  1.570798e+15   
2    9928.924808       4368.914092  1.570798e+15   
3   26908.978921      17856.472025  1.570798e+15   
4   26908.978921      17856.472025  1.570798e+15   

                                                s_t  label  \
0                (ts-ui-dashboard, ts-ui-dashboard)      0   
1           (istio-ingressgateway, ts-ui-dashboard)      0   
2  (ts-order-other-service, ts-order-other-service)      0   
3          (ts-station-service, ts-station-service)      0   
4      (ts-order-other-service, ts-station-service)      0   

                   source                  target  
0         ts-ui-dashboard         ts-ui-dashboard  
1    istio-ingressgateway         ts-ui-dashboard  
2  ts-order-other-service  ts-order-other-service  
3      ts-station-service      ts-station-service  
4  ts-order-other-service      ts-station-service  




select feature py
有一个方法输入异常数据和历史正常数据。
转化成np数组
计算历史的均值标准差，分别计算两组的α（减历史的均值，绝对值，除以历史的标准差）
判断两个结果的差距和阈值
- 用[source, target]作为索引
- 拿到输入和历史都有的indices，indices形如
    ('istio-ingressgateway', 'ts-assurance-service')
    ('istio-ingressgateway', 'ts-auth-service')
    ('istio-ingressgateway', 'ts-cancel-service')
    ('istio-ingressgateway', 'ts-consign-service')
- 遍历pair和所有的特征，如果特征有用添加到一个dict里，dict[(source, target)].append(feature)
- pkl序列化存储dict



anomaly detection
- 3sigma

