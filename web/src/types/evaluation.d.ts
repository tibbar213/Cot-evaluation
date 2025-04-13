export interface StrategyMetrics {
  total_records: number;
  metrics: {
    accuracy: {
      average_score: number;
      count: number;
    };
    reasoning_quality?: {
      average_score: number;
      count: number;
    };
  };
  difficulty_breakdown: {
    [key: string]: {
      count: number;
      accuracy: number;
    };
  };
  category_breakdown: {
    [key: string]: {
      count: number;
      accuracy: number;
    };
  };
} 