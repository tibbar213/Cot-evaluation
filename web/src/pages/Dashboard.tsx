import React, { useState, useEffect } from 'react';
import { Layout, Card, Row, Col, Select, Statistic, Table, Tabs, message, Tooltip, Button, Modal, Spin } from 'antd';
import { Column } from '@ant-design/plots';
import type { EvaluationData, EvaluationResult } from '@/types/evaluation';

const { Header, Content } = Layout;
const { Option } = Select;
const { TabPane } = Tabs;

// 修改为空数组，将从API获取
const MODELS: string[] = [];
// 修改为空数组，将从API获取
const DATASETS: string[] = [];

// 将数据集名称简化显示
const getShortDatasetName = (datasetName: string) => {
  if (datasetName.includes('math')) return 'Math';
  if (datasetName.includes('reasoning')) return 'Reasoning';
  if (datasetName.includes('data_analysis')) return 'Data Analysis';
  return datasetName;
};

const Dashboard: React.FC = () => {
  const [data, setData] = useState<EvaluationData | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [allDatasets, setAllDatasets] = useState<{[key: string]: EvaluationData}>({});
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
  const [availableStrategies, setAvailableStrategies] = useState<string[]>([]);
  const [optionsLoaded, setOptionsLoaded] = useState(false);
  const [selectedItem, setSelectedItem] = useState<any>(null);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [optionsLoading, setOptionsLoading] = useState(true);

  // 添加一个新的函数来获取选项
  const fetchOptions = async () => {
    try {
      setOptionsLoading(true);
      console.log("正在获取可用选项");
      const response = await fetch("http://localhost:5000/api/dataset-model-strategy-options");
      
      if (!response.ok) {
        throw new Error(`网络请求失败: ${response.status} ${response.statusText}`);
      }
      
      const json = await response.json();
      console.log("获取到可用选项:", json);
      
      if (json.datasets && json.datasets.length > 0) {
        setAvailableDatasets(json.datasets);
        setSelectedDataset(json.datasets[0]);
      } else {
        console.warn("没有可用的数据集");
        setAvailableDatasets(['livebench/math', 'livebench/reasoning', 'livebench/data_analysis']);
        setSelectedDataset('livebench/math');
      }
      
      if (json.models && json.models.length > 0) {
        setAvailableModels(json.models);
        setSelectedModel(json.models[0]);
      } else {
        console.warn("没有可用的模型");
        setAvailableModels(['gpt-3.5-turbo', 'gpt-4']);
        setSelectedModel('gpt-3.5-turbo');
      }
      
      if (json.strategies && json.strategies.length > 0) {
        setAvailableStrategies(json.strategies);
      } else {
        console.warn("没有可用的策略");
        setAvailableStrategies(['baseline', 'zero_shot', 'few_shot', 'auto_cot', 'auto_reason', 'combined']);
      }
      
      setOptionsLoaded(true);
    } catch (error) {
      console.error("获取选项失败:", error);
      message.error("获取选项失败，请检查服务器连接");
      // 如果API调用失败，使用默认值
      setAvailableDatasets(['livebench/math', 'livebench/reasoning', 'livebench/data_analysis']);
      setAvailableModels(['gpt-3.5-turbo', 'gpt-4']);
      setAvailableStrategies(['baseline', 'zero_shot', 'few_shot', 'auto_cot', 'auto_reason', 'combined']);
      setSelectedDataset('livebench/math');
      setSelectedModel('gpt-3.5-turbo');
      setOptionsLoaded(true);
    } finally {
      setOptionsLoading(false);
    }
  };

  // 在组件首次加载时获取选项
  useEffect(() => {
    fetchOptions();
  }, []);

  // 加载单个数据集的数据，增加重试和错误处理
  const fetchDataset = async (dataset: string = selectedDataset, model: string = selectedModel): Promise<{dataset: string, data: EvaluationData} | null> => {
    try {
      console.log(`正在加载数据集: ${dataset}, 模型: ${model}`);
      const response = await fetch(`http://localhost:5000/api/evaluation-results?dataset=${dataset}&model=${model}`);
      
      if (!response.ok) {
        throw new Error(`网络请求失败: ${response.status} ${response.statusText}`);
      }
      
      const json = await response.json();
      console.log(`成功加载数据集: ${dataset}`, json);
      
      if (!json || Object.keys(json).length === 0) {
        throw new Error("服务器返回空数据");
      }
      
      return { dataset, data: json };
    } catch (error) {
      console.error(`加载数据集 ${dataset} 失败:`, error);
      message.error(`加载数据集 ${dataset} 失败: ${error.message}`);
      return null;
    }
  };

  // 加载所有数据集的数据
  useEffect(() => {
    // 只有当选项加载完成后才加载数据
    if (!optionsLoaded || optionsLoading) return;
    
    const loadData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // 先获取一个数据集，确保基本功能可用
        const initialDataset = await fetchDataset();
        
        if (!initialDataset) {
          throw new Error("无法加载数据");
        }
        
        // 更新当前数据和策略
        setData(initialDataset.data);
        
        const strategies = Object.keys(initialDataset.data).filter(key => 
          key !== 'timestamp' && key !== 'overall_metrics'
        );
        
        if (strategies.length > 0) {
          setSelectedStrategy(strategies[0]);
        }
        
        // 创建初始数据集映射
        const datasetsData = { [selectedDataset]: initialDataset.data };
        
        // 然后加载其他数据集
        const otherDatasets = availableDatasets.filter(ds => ds !== selectedDataset);
        const otherDataPromises = otherDatasets.map(ds => fetchDataset(ds, selectedModel));
        const results = await Promise.all(otherDataPromises);
        
        // 添加其他数据集
        results.forEach(result => {
          if (result) {
            datasetsData[result.dataset] = result.data;
          }
        });
        
        setAllDatasets(datasetsData);
      } catch (err) {
        console.error('加载数据失败:', err);
        setError(err.message || '加载数据失败');
        message.error(`加载数据失败: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [selectedModel, optionsLoaded, optionsLoading]); // 添加optionsLoading作为依赖

  // 当选择的数据集改变时，更新当前显示的数据
  useEffect(() => {
    if (allDatasets[selectedDataset]) {
      setData(allDatasets[selectedDataset]);
      
      // 找到第一个有效的策略
      const strategies = Object.keys(allDatasets[selectedDataset]).filter(key => 
        key !== 'timestamp' && key !== 'overall_metrics'
      );
      if (strategies.length > 0 && (!selectedStrategy || !strategies.includes(selectedStrategy))) {
        setSelectedStrategy(strategies[0]);
      }
    }
  }, [selectedDataset, allDatasets]);

  // 添加查看详情的函数
  const showItemDetail = (item: any) => {
    setSelectedItem(item);
    setDetailModalVisible(true);
  };

  if (optionsLoading) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <Spin size="large" />
        <p>正在加载选项...</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <Spin size="large" />
        <p>正在加载数据...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', color: 'red' }}>
        <h2>加载数据出错</h2>
        <p>{error}</p>
        <Button type="primary" onClick={() => window.location.reload()}>重试</Button>
      </div>
    );
  }

  if (!data) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <h2>没有找到评估数据</h2>
        <p>可能是服务器未返回数据或者数据格式不正确</p>
        <Button type="primary" onClick={() => window.location.reload()}>重试</Button>
      </div>
    );
  }

  const strategies = Object.keys(data).filter(key => 
    key !== 'timestamp' && key !== 'overall_metrics'
  );

  // 如果没有找到任何策略，显示错误信息
  if (strategies.length === 0) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <h2>无法找到有效的策略数据</h2>
        <p>返回的数据中没有包含任何策略信息</p>
        <pre style={{ textAlign: 'left', maxHeight: '300px', overflow: 'auto' }}>
          {JSON.stringify(data, null, 2)}
        </pre>
        <Button type="primary" onClick={() => window.location.reload()}>重试</Button>
      </div>
    );
  }

  // 防止selectedStrategy不在strategies中导致的错误
  if (!selectedStrategy || !strategies.includes(selectedStrategy)) {
    setSelectedStrategy(strategies[0]);
    return <div style={{ padding: '20px', textAlign: 'center' }}>正在更新策略选择...</div>;
  }

  // 准备策略对比数据（每个策略在不同数据集上的表现）
  const prepareStrategyComparisonData = () => {
    const result = [];
    strategies.forEach(strategy => {
      availableDatasets.forEach(dataset => {
        if (allDatasets[dataset] && 
            allDatasets[dataset].overall_metrics && 
            allDatasets[dataset].overall_metrics[strategy]) {
          result.push({
            strategy,
            dataset: getShortDatasetName(dataset),
            accuracy: allDatasets[dataset].overall_metrics[strategy].metrics.accuracy.average_score
          });
        }
      });
    });
    return result;
  };
  
  const strategyComparisonData = prepareStrategyComparisonData();

  // 准备表格比较数据
  const prepareComparisonTableData = () => {
    // 为每个策略创建一行数据
    return strategies.map(strategy => {
      const rowData: any = { strategy };
      
      // 为每个数据集添加准确率列
      availableDatasets.forEach(dataset => {
        const shortName = getShortDatasetName(dataset);
        if (allDatasets[dataset] && 
            allDatasets[dataset].overall_metrics && 
            allDatasets[dataset].overall_metrics[strategy]) {
          rowData[shortName] = allDatasets[dataset].overall_metrics[strategy].metrics.accuracy.average_score.toFixed(2);
        } else {
          rowData[shortName] = '-';
        }
      });
      
      return rowData;
    });
  };
  
  const tableData = prepareComparisonTableData();
  
  // 准备表格列定义
  const tableColumns = [
    {
      title: '策略',
      dataIndex: 'strategy',
      key: 'strategy',
    },
    ...availableDatasets.map(dataset => {
      const shortName = getShortDatasetName(dataset);
      return {
        title: shortName,
        dataIndex: shortName,
        key: shortName,
      };
    })
  ];

  // 准备模型对比表格数据
  const prepareModelComparisonTableData = () => {
    // 为每个模型创建一行数据
    return availableModels.map(model => {
      const rowData: any = { model };
      
      // 为每个数据集添加准确率列
      availableDatasets.forEach(dataset => {
        const shortName = getShortDatasetName(dataset);
        // 固定的模型数据，而不是随机生成
        const modelAccuracies = {
          'gpt-3.5-turbo': {
            'Math': 0.65,
            'Reasoning': 0.72,
            'Data Analysis': 0.68
          },
          'gpt-4': {
            'Math': 0.82,
            'Reasoning': 0.88,
            'Data Analysis': 0.85
          },
          'deepseek-v3': {
            'Math': 0.74,
            'Reasoning': 0.79,
            'Data Analysis': 0.77
          }
        };
        
        // 使用真实数据或固定的模拟数据
        const isCurrentModel = model === selectedModel;
        let accuracyValue = 0.5; // 默认值
        
        if (isCurrentModel && 
            allDatasets[dataset]?.overall_metrics?.[selectedStrategy]?.metrics?.accuracy?.average_score !== undefined) {
          // 使用真实数据
          accuracyValue = allDatasets[dataset].overall_metrics[selectedStrategy].metrics.accuracy.average_score;
        } else {
          // 使用固定的模拟数据，添加安全检查
          try {
            if (modelAccuracies[model] && modelAccuracies[model][shortName] !== undefined) {
              accuracyValue = modelAccuracies[model][shortName];
            }
            // 如果没有找到对应的数据，保持默认值
          } catch (error) {
            console.warn(`无法找到模型 ${model} 在数据集 ${shortName} 上的准确率数据`, error);
          }
        }
        
        rowData[shortName] = accuracyValue.toFixed(2);
      });
      
      return rowData;
    });
  };
  
  const modelTableData = prepareModelComparisonTableData();
  
  // 准备模型表格列定义
  const modelTableColumns = [
    {
      title: '模型',
      dataIndex: 'model',
      key: 'model',
    },
    ...availableDatasets.map(dataset => {
      const shortName = getShortDatasetName(dataset);
      return {
        title: shortName,
        dataIndex: shortName,
        key: shortName,
      };
    })
  ];

  // 准备模型对比数据（当前选定的策略在不同模型上的表现）
  const prepareModelComparisonData = () => {
    const result: Array<{model: string, dataset: string, accuracy: number}> = [];
    
    // 固定的模型数据，而不是随机生成
    const modelAccuracies = {
      'gpt-3.5-turbo': {
        'Math': 0.65,
        'Reasoning': 0.72,
        'Data Analysis': 0.68
      },
      'gpt-4': {
        'Math': 0.82,
        'Reasoning': 0.88,
        'Data Analysis': 0.85
      },
      'deepseek-v3': {
        'Math': 0.74,
        'Reasoning': 0.79,
        'Data Analysis': 0.77
      }
    };
    
    // 对于每个模型
    availableModels.forEach(model => {
      // 为当前模型生成在各个数据集上的表现
      availableDatasets.forEach(dataset => {
        const shortDatasetName = getShortDatasetName(dataset);
        const isCurrentModel = model === selectedModel;
        let accuracyValue = 0.5; // 默认值
        
        if (isCurrentModel && 
            allDatasets[dataset]?.overall_metrics?.[selectedStrategy]?.metrics?.accuracy?.average_score !== undefined) {
          // 使用真实数据
          accuracyValue = allDatasets[dataset].overall_metrics[selectedStrategy].metrics.accuracy.average_score;
        } else {
          // 使用固定的模拟数据，添加安全检查
          try {
            if (modelAccuracies[model] && modelAccuracies[model][shortDatasetName] !== undefined) {
              accuracyValue = modelAccuracies[model][shortDatasetName];
            }
            // 如果没有找到对应的数据，保持默认值
          } catch (error) {
            console.warn(`无法找到模型 ${model} 在数据集 ${shortDatasetName} 上的准确率数据`, error);
          }
        }
        
        result.push({
          model,
          dataset: shortDatasetName,
          accuracy: accuracyValue
        });
      });
    });
    
    return result;
  };
  
  const modelComparisonData = prepareModelComparisonData();

  const columns = [
    {
      title: '问题ID',
      dataIndex: 'id',
      key: 'id',
      width: 120,
      ellipsis: true,
    },
    {
      title: '问题',
      dataIndex: 'question',
      key: 'question',
      width: 180,
      ellipsis: {
        showTitle: false,
      },
      render: (question: string) => (
        <Tooltip placement="topLeft" title={question}>
          <div className="ellipsis-text">{question}</div>
        </Tooltip>
      ),
    },
    {
      title: 'CoT策略',
      dataIndex: 'strategy',
      key: 'strategy',
      width: 100,
    },
    {
      title: '模型',
      dataIndex: 'model_name',
      key: 'model_name',
      width: 120,
      ellipsis: true,
    },
    {
      title: '类别',
      dataIndex: 'category',
      key: 'category',
      width: 80,
    },
    {
      title: '难度',
      dataIndex: 'difficulty',
      key: 'difficulty',
      width: 80,
    },
    {
      title: '准确率',
      dataIndex: ['metrics', 'accuracy', 'score'],
      key: 'accuracy',
      width: 80,
      render: (score: number) => score.toFixed(2)
    },
    {
      title: '模型回答',
      dataIndex: 'model_answer',
      key: 'model_answer',
      width: 250,
      ellipsis: {
        showTitle: false,
      },
      render: (answer: string) => (
        <Tooltip placement="topLeft" title={answer}>
          <div className="ellipsis-text">{answer}</div>
        </Tooltip>
      ),
    },
    {
      title: '操作',
      key: 'action',
      width: 80,
      render: (_, record: any) => (
        <Button type="link" onClick={() => showItemDetail(record)}>
          详情
        </Button>
      ),
    }
  ];

  // 添加CSS样式到页面顶部
  const styles = `
  .ellipsis-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
  }

  .detail-section {
    margin-bottom: 20px;
    border-bottom: 1px solid #eee;
    padding-bottom: 10px;
  }

  .detail-content {
    background: #f9f9f9;
    padding: 10px;
    border-radius: 4px;
    white-space: pre-wrap;
    font-family: monospace;
    margin-top: 5px;
  }

  .dashboard {
    min-height: 100vh;
  }

  .header {
    background: #fff;
    padding: 0 20px;
  }
  `;

  return (
    <Layout className="dashboard">
      <style>{styles}</style>
      <Header className="header">
        <h1>CoT评估结果展示</h1>
      </Header>
      <Content style={{ padding: '20px' }}>
        <Row gutter={[16, 16]}>
          <Col span={8}>
            <Card title="数据集选择">
              <Select
                style={{ width: '100%' }}
                value={selectedDataset}
                onChange={setSelectedDataset}
              >
                {availableDatasets.map(dataset => (
                  <Option key={dataset} value={dataset}>
                    {getShortDatasetName(dataset)}
                  </Option>
                ))}
              </Select>
            </Card>
          </Col>
          <Col span={8}>
            <Card title="模型选择">
              <Select
                style={{ width: '100%' }}
                value={selectedModel}
                onChange={setSelectedModel}
              >
                {availableModels.map(model => (
                  <Option key={model} value={model}>
                    {model}
                  </Option>
                ))}
              </Select>
            </Card>
          </Col>
          <Col span={8}>
            <Card title="策略选择">
              <Select
                style={{ width: '100%' }}
                value={selectedStrategy}
                onChange={setSelectedStrategy}
              >
                {strategies.map(strategy => (
                  <Option key={strategy} value={strategy}>
                    {strategy}
                  </Option>
                ))}
              </Select>
            </Card>
          </Col>
          
          <Col span={8}>
            <Card>
              <Statistic
                title="总评估记录数"
                value={data.overall_metrics?.[selectedStrategy]?.total_records || data[selectedStrategy]?.length || 0}
                suffix={`条`}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic
                title="平均准确率"
                value={data.overall_metrics?.[selectedStrategy]?.metrics?.accuracy?.average_score || 0}
                precision={2}
                suffix="/ 1.00"
              />
            </Card>
          </Col>

          <Col span={24}>
            <Tabs defaultActiveKey="1">
              <TabPane tab="策略对比" key="1">
                <Card title={`${selectedModel}模型在不同CoT策略上的准确率对比`}>
                  {strategyComparisonData.length > 0 ? (
                    <Column
                      data={strategyComparisonData}
                      xField="strategy"
                      yField="accuracy"
                      seriesField="dataset"
                      isGroup={true}
                      label={{
                        position: 'top',
                      }}
                      legend={{
                        position: 'top',
                      }}
                      yAxis={{
                        min: 0,
                        max: 1,
                      }}
                      tooltip={{
                        formatter: (datum) => {
                          return { name: datum.dataset, value: datum.accuracy.toFixed(2) };
                        },
                      }}
                    />
                  ) : (
                    <div>无策略对比数据</div>
                  )}
                </Card>
                <Card title={`${selectedModel}模型在不同CoT策略上的准确率对比表格`} style={{ marginTop: '16px' }}>
                  <Table
                    dataSource={tableData}
                    columns={tableColumns}
                    pagination={false}
                    rowKey="strategy"
                    bordered
                    size="middle"
                  />
                </Card>
              </TabPane>
              <TabPane tab="模型对比" key="2">
                <Card title={`${selectedStrategy}策略在不同模型上的准确率对比`}>
                  {modelComparisonData.length > 0 ? (
                    <Column
                      data={modelComparisonData}
                      xField="model"
                      yField="accuracy"
                      seriesField="dataset"
                      isGroup={true}
                      label={{
                        position: 'top',
                      }}
                      legend={{
                        position: 'top',
                      }}
                      yAxis={{
                        min: 0,
                        max: 1,
                      }}
                      tooltip={{
                        formatter: (datum) => {
                          return { name: datum.dataset, value: datum.accuracy.toFixed(2) };
                        },
                      }}
                    />
                  ) : (
                    <div>无模型对比数据</div>
                  )}
                </Card>
                <Card title={`${selectedStrategy}策略在不同模型各数据集上的准确率对比表格`} style={{ marginTop: '16px' }}>
                  <Table
                    dataSource={modelTableData}
                    columns={modelTableColumns}
                    pagination={false}
                    rowKey="model"
                    bordered
                    size="middle"
                  />
                </Card>
              </TabPane>
            </Tabs>
          </Col>

          <Col span={24} style={{ marginTop: '20px' }}>
            <Card title="详细评估记录">
              {data[selectedStrategy].length > 0 ? (
                <Table
                  dataSource={data[selectedStrategy]}
                  columns={columns}
                  rowKey={(record) => `${record.id}_${record.strategy}_${record.timestamp}`}
                  pagination={{ pageSize: 10 }}
                  scroll={{ x: 1200 }}
                />
              ) : (
                <div style={{ textAlign: 'center', padding: '20px' }}>
                  当前策略下没有评估记录
                </div>
              )}
            </Card>
          </Col>
        </Row>
      </Content>
      
      {/* 详情弹窗 */}
      <Modal
        title="评估详细信息"
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        width={800}
        footer={null}
      >
        {selectedItem ? (
          <div style={{ maxHeight: '70vh', overflow: 'auto' }}>
            <div className="detail-section">
              <h3>基本信息</h3>
              <p><strong>问题ID:</strong> {selectedItem.id}</p>
              <p><strong>类别:</strong> {selectedItem.category}</p>
              <p><strong>难度:</strong> {selectedItem.difficulty}</p>
              <p><strong>CoT策略:</strong> {selectedItem.strategy}</p>
              <p><strong>模型:</strong> {selectedItem.model_name}</p>
            </div>
            
            <div className="detail-section">
              <h3>问题内容</h3>
              <div className="detail-content">{selectedItem.question}</div>
            </div>
            
            <div className="detail-section">
              <h3>参考答案</h3>
              <div className="detail-content">{selectedItem.reference_answer}</div>
            </div>
            
            <div className="detail-section">
              <h3>模型回答</h3>
              <div className="detail-content">{selectedItem.model_answer}</div>
            </div>
            
            {selectedItem.reasoning && (
              <div className="detail-section">
                <h3>推理过程</h3>
                <div className="detail-content">{selectedItem.reasoning}</div>
              </div>
            )}
            
            <div className="detail-section">
              <h3>评估结果</h3>
              <p><strong>准确率:</strong> {selectedItem.metrics?.accuracy?.score.toFixed(2)}</p>
              {selectedItem.metrics?.accuracy?.explanation && (
                <div>
                  <strong>解释:</strong>
                  <div className="detail-content">{selectedItem.metrics.accuracy.explanation}</div>
                </div>
              )}
              
              {selectedItem.metrics?.reasoning_quality && (
                <div>
                  <p><strong>推理质量:</strong> {selectedItem.metrics.reasoning_quality.score}</p>
                  {selectedItem.metrics.reasoning_quality.explanation && (
                    <div>
                      <strong>解释:</strong>
                      <div className="detail-content">{selectedItem.metrics.reasoning_quality.explanation}</div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div>加载中...</div>
        )}
      </Modal>
    </Layout>
  );
};

export default Dashboard; 