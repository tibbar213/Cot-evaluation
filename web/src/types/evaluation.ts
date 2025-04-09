export interface EvaluationMetrics {
  accuracy: {
    score: number;
    explanation: string;
  };
  reasoning_quality: {
    score: number;
    explanation: string;
  };
}

export interface EvaluationResult {
  id: string;
  question: string;
  reference_answer: string;
  model_answer: string;
  reasoning: string;
  category: string;
  difficulty: string;
  metrics: EvaluationMetrics;
  timestamp: number;
}

export interface StrategyResults {
  [strategy: string]: EvaluationResult[];
}

export interface DifficultyBreakdown {
  [difficulty: string]: { count: number; accuracy: number };
}

export interface CategoryBreakdown {
  [category: string]: { count: number; accuracy: number };
}

export interface StrategyMetrics {
  total_questions: number;
  metrics: {
    accuracy: {
      average_score: number;
      count: number;
    };
    reasoning_quality: {
      average_score: number;
      count: number;
    };
  };
  difficulty_breakdown: DifficultyBreakdown;
  category_breakdown: CategoryBreakdown;
}

export interface OverallMetrics {
  [strategy: string]: StrategyMetrics;
}

export interface EvaluationData {
  [strategy: string]: EvaluationResult[];
  timestamp: number;
  overall_metrics: OverallMetrics;
} 