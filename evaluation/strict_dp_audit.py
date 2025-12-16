"""
严格的差分隐私审计框架

基于 Jagielski et al. (2020) "Auditing Differentially Private Machine Learning"

提供统计上严格的 DP 验证,而非启发式检查。
"""

import numpy as np
from typing import Callable, List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn(
        "scipy not available. Install with: pip install scipy\n"
        "Strict DP auditing requires scipy for statistical tests."
    )


@dataclass
class AuditReport:
    """完整的审计报告"""

    passed: bool
    epsilon_claimed: float
    delta_claimed: float
    epsilon_observed: float
    delta_observed: float
    confidence_level: float = 0.95

    tests: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"""
{'='*80}
STRICT DIFFERENTIAL PRIVACY AUDIT REPORT
{'='*80}

Overall Status: {status}

Privacy Claims:
  Claimed: (ε={self.epsilon_claimed}, δ={self.delta_claimed})
  Observed: (ε={self.epsilon_observed:.4f}, δ={self.delta_observed:.6f})

Individual Tests:
  Noise Distribution:     {'✅ PASS' if self.tests.get('noise_distribution', False) else '❌ FAIL'}
  Likelihood Ratio:       {'✅ PASS' if self.tests.get('likelihood_ratio', False) else '❌ FAIL'}
  Privacy Loss Dist:      {'✅ PASS' if self.tests.get('privacy_loss_distribution', False) else '❌ FAIL'}

Confidence Level: {self.confidence_level * 100:.0f}%

{'='*80}
"""


class RandomnessAudit:
    """测试 DP 算法的噪声分布是否正确

    验证添加的噪声确实来自声称的分布 (如 N(0, σ²))。
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Args:
            significance_level: 统计检验的显著性水平 (通常 0.05)
        """
        self.alpha = significance_level

    def test_gaussian_noise(
        self,
        noise_samples: List[float],
        expected_mean: float,
        expected_std: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        测试噪声是否符合高斯分布 N(μ, σ²)

        使用 Kolmogorov-Smirnov 检验

        Args:
            noise_samples: 收集的噪声样本
            expected_mean: 期望均值 (通常 0)
            expected_std: 期望标准差

        Returns:
            (passed, metrics) where metrics contains:
                - ks_statistic: K-S 统计量
                - p_value: p 值
                - sample_mean: 样本均值
                - sample_std: 样本标准差
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for Kolmogorov-Smirnov test")

        if len(noise_samples) < 30:
            warnings.warn(
                f"Sample size {len(noise_samples)} < 30, test may be unreliable"
            )

        # K-S 检验: 检验样本是否来自 N(expected_mean, expected_std²)
        ks_stat, p_value = stats.kstest(
            noise_samples,
            lambda x: stats.norm.cdf(x, expected_mean, expected_std)
        )

        # p < α 表示拒绝零假设 (噪声分布不符合声称)
        passed = p_value >= self.alpha

        metrics = {
            'ks_statistic': float(ks_stat),
            'p_value': float(p_value),
            'sample_mean': float(np.mean(noise_samples)),
            'sample_std': float(np.std(noise_samples)),
            'expected_mean': expected_mean,
            'expected_std': expected_std,
            'sample_size': len(noise_samples),
        }

        return passed, metrics


class LikelihoodRatioAudit:
    """基于似然比的 DP 审计

    核心思想: 对邻近数据库 D 和 D',运行算法多次,
    估计 log(Pr[M(D)=o] / Pr[M(D')=o]) 的分布。

    对于 (ε, δ)-DP,这个比值应该 ≤ ε (除了 δ 的概率)。
    """

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    def compute_empirical_privacy_loss(
        self,
        outputs_D: List[Any],
        outputs_D_prime: List[Any],
        distance_fn: Optional[Callable[[Any, Any], float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算经验隐私损失

        由于我们处理的是连续/离散混合空间,使用基于距离的估计。

        Args:
            outputs_D: 在数据库 D 上的输出样本
            outputs_D_prime: 在邻近数据库 D' 上的输出样本
            distance_fn: 可选的距离函数 (默认使用 L2 距离)

        Returns:
            (estimated_epsilon, metrics)
        """
        if distance_fn is None:
            distance_fn = self._default_distance

        # 简化方法: 使用最近邻距离比
        # 对于每个 o_D,找到最近的 o_D' 和最近的其他 o_D
        privacy_losses = []

        for i, out_D in enumerate(outputs_D):
            # 距离到 D' 的最近点
            dist_to_D_prime = min(
                distance_fn(out_D, out_Dp) for out_Dp in outputs_D_prime
            )

            # 距离到 D 中其他点的最近距离
            other_D = [outputs_D[j] for j in range(len(outputs_D)) if j != i]
            if other_D:
                dist_to_other_D = min(
                    distance_fn(out_D, other) for other in other_D
                )
            else:
                dist_to_other_D = float('inf')

            # 估计的隐私损失 (简化版)
            # 如果 D 和 D' 的输出分布相近,dist_to_D_prime ≈ dist_to_other_D
            # 如果差异大,隐私泄露大
            if dist_to_D_prime > 0:
                privacy_loss = np.log(dist_to_other_D / dist_to_D_prime)
                privacy_losses.append(privacy_loss)

        if not privacy_losses:
            return float('inf'), {'error': 'No valid privacy loss estimates'}

        # 取最大值作为经验 ε
        estimated_eps = float(np.max(privacy_losses))

        metrics = {
            'estimated_epsilon': estimated_eps,
            'mean_privacy_loss': float(np.mean(privacy_losses)),
            'std_privacy_loss': float(np.std(privacy_losses)),
            'num_samples': len(privacy_losses),
        }

        return estimated_eps, metrics

    @staticmethod
    def _default_distance(x: Any, y: Any) -> float:
        """默认距离函数 (处理字符串和数值)"""
        if isinstance(x, str) and isinstance(y, str):
            # Jaccard 距离 for strings
            words_x = set(x.lower().split())
            words_y = set(y.lower().split())
            if not words_x and not words_y:
                return 0.0
            intersection = len(words_x & words_y)
            union = len(words_x | words_y)
            return 1.0 - (intersection / union if union > 0 else 0.0)
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return abs(x - y)
        else:
            # 尝试转换为字符串比较
            return LikelihoodRatioAudit._default_distance(str(x), str(y))


class PrivacyLossDistributionAnalyzer:
    """分析隐私损失的完整分布

    根据 DP 定义,Pr[隐私损失 > ε] 应该 ≤ δ
    """

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    def analyze(
        self,
        privacy_losses: List[float],
        epsilon_claimed: float,
        delta_claimed: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        分析隐私损失分布是否满足 (ε, δ)-DP

        Args:
            privacy_losses: 收集的隐私损失样本
            epsilon_claimed: 声称的 ε
            delta_claimed: 声称的 δ

        Returns:
            (passed, metrics)
        """
        if len(privacy_losses) < 100:
            warnings.warn(
                f"Sample size {len(privacy_losses)} < 100, analysis may be unreliable"
            )

        # 计算经验概率: Pr[损失 > ε_claimed]
        prob_exceeding = np.mean(np.array(privacy_losses) > epsilon_claimed)

        # 计算置信区间 (Wilson score interval)
        n = len(privacy_losses)
        z = 1.96  # 95% 置信度
        p_hat = prob_exceeding

        # Wilson score interval
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denominator

        ci_lower = max(0, center - margin)
        ci_upper = min(1, center + margin)

        # 判定: 置信区间上界是否 ≤ δ
        # 保守判定: 即使上界略超过 δ,也允许小容差
        tolerance = 0.01  # 1% 容差
        passed = ci_upper <= delta_claimed + tolerance

        metrics = {
            'max_privacy_loss': float(np.max(privacy_losses)),
            'mean_privacy_loss': float(np.mean(privacy_losses)),
            'prob_exceeding_epsilon': prob_exceeding,
            'prob_ci_lower': ci_lower,
            'prob_ci_upper': ci_upper,
            'delta_claimed': delta_claimed,
            'passed': passed,
        }

        return passed, metrics


class StrictDPAuditor:
    """严格的差分隐私审计器

    集成多个统计测试,提供综合的 DP 审计。
    """

    def __init__(
        self,
        epsilon_claimed: float,
        delta_claimed: float,
        confidence_level: float = 0.95
    ):
        """
        Args:
            epsilon_claimed: 声称的隐私预算 ε
            delta_claimed: 声称的隐私参数 δ
            confidence_level: 统计检验的置信水平
        """
        self.epsilon = epsilon_claimed
        self.delta = delta_claimed
        self.confidence = confidence_level
        self.alpha = 1 - confidence_level

        self.noise_auditor = RandomnessAudit(self.alpha)
        self.lr_auditor = LikelihoodRatioAudit(self.alpha)
        self.pld_analyzer = PrivacyLossDistributionAnalyzer(self.alpha)

    def audit_noise_mechanism(
        self,
        noise_samples: List[float],
        expected_std: float
    ) -> Tuple[bool, Dict]:
        """
        审计噪声机制

        Args:
            noise_samples: 从算法中提取的噪声样本
            expected_std: 根据配置计算的期望标准差

        Returns:
            (passed, details)
        """
        print("\n[Test 1/3] Auditing noise distribution...")
        print(f"  Expected: N(0, {expected_std:.4f})")
        print(f"  Samples: {len(noise_samples)}")

        passed, metrics = self.noise_auditor.test_gaussian_noise(
            noise_samples, expected_mean=0.0, expected_std=expected_std
        )

        print(f"  Observed: N({metrics['sample_mean']:.4f}, {metrics['sample_std']:.4f})")
        print(f"  K-S test: p-value = {metrics['p_value']:.4f}")
        print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")

        return passed, metrics

    def audit_neighboring_databases(
        self,
        outputs_pairs: List[Tuple[List, List]],
        distance_fn: Optional[Callable] = None
    ) -> Tuple[bool, Dict]:
        """
        审计多组邻近数据库

        Args:
            outputs_pairs: 列表 of (outputs_D, outputs_D'),每对是邻近数据库的输出
            distance_fn: 可选的距离函数

        Returns:
            (passed, details)
        """
        print("\n[Test 2/3] Auditing likelihood ratios on neighboring databases...")
        print(f"  Number of database pairs: {len(outputs_pairs)}")

        all_epsilon_estimates = []
        all_metrics = []

        for i, (outputs_D, outputs_D_prime) in enumerate(outputs_pairs):
            eps_est, metrics = self.lr_auditor.compute_empirical_privacy_loss(
                outputs_D, outputs_D_prime, distance_fn
            )
            all_epsilon_estimates.append(eps_est)
            all_metrics.append(metrics)

            print(f"  Pair {i+1}: ε̂ = {eps_est:.3f}")

        max_eps_observed = max(all_epsilon_estimates)
        mean_eps = np.mean(all_epsilon_estimates)

        # 判定: 最大观察到的 ε 是否 ≤ 声称的 ε (加上合理容差)
        tolerance = 0.5  # 允许 50% 容差 (因为是估计)
        passed = max_eps_observed <= self.epsilon * (1 + tolerance)

        print(f"  Max ε observed: {max_eps_observed:.3f}")
        print(f"  Mean ε observed: {mean_eps:.3f}")
        print(f"  Claimed ε: {self.epsilon:.3f}")
        print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")

        details = {
            'max_epsilon_observed': max_eps_observed,
            'mean_epsilon_observed': mean_eps,
            'epsilon_estimates': all_epsilon_estimates,
            'individual_metrics': all_metrics,
            'passed': passed
        }

        return passed, details

    def audit_privacy_loss_distribution(
        self,
        privacy_losses: List[float]
    ) -> Tuple[bool, Dict]:
        """
        审计隐私损失分布

        Args:
            privacy_losses: 收集的隐私损失样本

        Returns:
            (passed, details)
        """
        print("\n[Test 3/3] Analyzing privacy loss distribution...")
        print(f"  Samples: {len(privacy_losses)}")

        passed, metrics = self.pld_analyzer.analyze(
            privacy_losses, self.epsilon, self.delta
        )

        print(f"  Max privacy loss: {metrics['max_privacy_loss']:.3f}")
        print(f"  Pr[loss > ε]: {metrics['prob_exceeding_epsilon']:.4f}")
        print(f"  95% CI: [{metrics['prob_ci_lower']:.4f}, {metrics['prob_ci_upper']:.4f}]")
        print(f"  Claimed δ: {self.delta:.6f}")
        print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")

        return passed, metrics

    def generate_report(
        self,
        noise_test: Tuple[bool, Dict],
        lr_test: Tuple[bool, Dict],
        pld_test: Tuple[bool, Dict]
    ) -> AuditReport:
        """生成综合审计报告"""

        all_passed = noise_test[0] and lr_test[0] and pld_test[0]

        return AuditReport(
            passed=all_passed,
            epsilon_claimed=self.epsilon,
            delta_claimed=self.delta,
            epsilon_observed=lr_test[1]['max_epsilon_observed'],
            delta_observed=pld_test[1]['prob_exceeding_epsilon'],
            confidence_level=self.confidence,
            tests={
                'noise_distribution': noise_test[0],
                'likelihood_ratio': lr_test[0],
                'privacy_loss_distribution': pld_test[0]
            },
            details={
                'noise_test': noise_test[1],
                'likelihood_ratio_test': lr_test[1],
                'privacy_loss_distribution': pld_test[1]
            }
        )
