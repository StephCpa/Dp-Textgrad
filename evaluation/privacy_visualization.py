"""
Privacy Verification Visualization Tools

Generate visual reports for privacy verification test results.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization disabled.")


class PrivacyVisualizationReport:
    """Generate visual reports from privacy verification results."""

    def __init__(self, results_dir: str = "privacy_verification_results"):
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / "verification_results.json"

        if not self.results_file.exists():
            raise FileNotFoundError(
                f"Results file not found: {self.results_file}\n"
                "Run example_privacy_verification.py first to generate results."
            )

        with open(self.results_file, 'r') as f:
            self.data = json.load(f)

    def generate_summary_plot(self, output_file: str = None):
        """Generate overview summary plot."""

        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Skipping visualization.")
            return

        results = self.data["results"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Privacy Verification Summary", fontsize=16, fontweight='bold')

        # 1. Test Pass/Fail Overview
        ax1 = axes[0, 0]
        test_names = [r["test_name"] for r in results]
        pass_status = [1 if r["passed"] else 0 for r in results]
        colors = ['green' if p else 'red' for p in pass_status]

        bars = ax1.barh(range(len(test_names)), [1]*len(test_names), color=colors, alpha=0.6)
        ax1.set_yticks(range(len(test_names)))
        ax1.set_yticklabels([f"{i+1}. {name[:30]}..." if len(name) > 30 else f"{i+1}. {name}"
                             for i, name in enumerate(test_names)], fontsize=9)
        ax1.set_xlabel('Status', fontsize=10)
        ax1.set_title('Test Results Overview', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 1.2)
        ax1.set_xticks([])

        # Add pass/fail labels
        for i, (bar, passed) in enumerate(zip(bars, pass_status)):
            label = "✓ PASS" if passed else "✗ FAIL"
            color = 'darkgreen' if passed else 'darkred'
            ax1.text(0.5, i, label, ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

        # 2. Privacy Budget Distribution
        ax2 = axes[0, 1]
        epsilons = [r["epsilon"] for r in results if "epsilon" in r]
        test_labels = [r["test_name"][:20] for r in results if "epsilon" in r]

        if epsilons:
            bars2 = ax2.bar(range(len(epsilons)), epsilons, color='steelblue', alpha=0.7)
            ax2.set_xticks(range(len(epsilons)))
            ax2.set_xticklabels(range(1, len(epsilons) + 1))
            ax2.set_xlabel('Test Number', fontsize=10)
            ax2.set_ylabel('Epsilon (ε)', fontsize=10)
            ax2.set_title('Privacy Budget per Test', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar, eps in zip(bars2, epsilons):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{eps:.2f}', ha='center', va='bottom', fontsize=9)

        # 3. Membership Inference Attack Success Rate
        ax3 = axes[1, 0]
        mia_result = next((r for r in results if "Membership Inference" in r["test_name"]), None)

        if mia_result and "attack_success_rate" in mia_result["metrics"]:
            success_rate = mia_result["metrics"]["attack_success_rate"]
            theoretical_baseline = 0.5

            categories = ['Random\nGuessing', 'Observed\nAttack Rate']
            values = [theoretical_baseline, success_rate]
            colors_mia = ['gray', 'orange' if success_rate > 0.6 else 'green']

            bars3 = ax3.bar(categories, values, color=colors_mia, alpha=0.7)
            ax3.set_ylabel('Success Rate', fontsize=10)
            ax3.set_title('Membership Inference Attack Resistance', fontsize=12, fontweight='bold')
            ax3.set_ylim(0, 1)
            ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax3.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar, val in zip(bars3, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Add interpretation
            if success_rate <= 0.55:
                interpretation = "✓ Strong resistance"
                color = 'green'
            elif success_rate <= 0.65:
                interpretation = "⚠ Moderate resistance"
                color = 'orange'
            else:
                interpretation = "✗ Weak resistance"
                color = 'red'

            ax3.text(0.5, 0.95, interpretation, transform=ax3.transAxes,
                    ha='center', va='top', fontsize=10, fontweight='bold',
                    color=color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 4. Noise Distribution Validation
        ax4 = axes[1, 1]
        noise_result = next((r for r in results if "Noise Distribution" in r["test_name"]), None)

        if noise_result and "observed_std" in noise_result["metrics"]:
            observed = noise_result["metrics"]["observed_std"]
            expected = noise_result["metrics"]["expected_std"]

            categories = ['Expected σ', 'Observed σ']
            values = [expected, observed]
            colors_noise = ['steelblue', 'green' if abs(observed - expected) / expected < 0.2 else 'orange']

            bars4 = ax4.bar(categories, values, color=colors_noise, alpha=0.7)
            ax4.set_ylabel('Standard Deviation', fontsize=10)
            ax4.set_title('Noise Distribution Validation', fontsize=12, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar, val in zip(bars4, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)

            # Add error percentage
            error_pct = abs(observed - expected) / expected * 100
            ax4.text(0.5, 0.95, f'Error: {error_pct:.1f}%',
                    transform=ax4.transAxes, ha='center', va='top',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if output_file is None:
            output_file = self.results_dir / "privacy_verification_summary.png"
        else:
            output_file = Path(output_file)

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Summary plot saved to: {output_file}")

        return fig

    def generate_detailed_report(self, output_file: str = None):
        """Generate detailed multi-page report."""

        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Skipping visualization.")
            return

        results = self.data["results"]

        # Create figure with subplots for each test
        n_tests = len(results)
        fig = plt.figure(figsize=(16, 4 * n_tests))
        fig.suptitle("Detailed Privacy Verification Report", fontsize=18, fontweight='bold', y=0.995)

        for idx, result in enumerate(results):
            # Create subplot for this test
            ax = plt.subplot(n_tests, 3, idx * 3 + 1)

            test_name = result["test_name"]
            passed = result["passed"]
            metrics = result["metrics"]

            # Title with pass/fail
            status_color = 'green' if passed else 'red'
            status_text = "✓ PASSED" if passed else "✗ FAILED"
            ax.text(0.5, 0.9, f"{idx + 1}. {test_name}",
                   ha='center', va='top', fontsize=12, fontweight='bold',
                   transform=ax.transAxes)
            ax.text(0.5, 0.7, status_text,
                   ha='center', va='top', fontsize=14, fontweight='bold',
                   color=status_color, transform=ax.transAxes)

            # Display key metrics
            y_pos = 0.5
            for key, value in list(metrics.items())[:5]:  # Show top 5 metrics
                if value is not None:
                    if isinstance(value, float):
                        text = f"{key}: {value:.4f}"
                    else:
                        text = f"{key}: {value}"
                    ax.text(0.1, y_pos, text, ha='left', va='top',
                           fontsize=9, transform=ax.transAxes,
                           family='monospace')
                    y_pos -= 0.12

            ax.axis('off')

            # Metrics visualization
            if "Membership Inference" in test_name:
                ax2 = plt.subplot(n_tests, 3, idx * 3 + 2)
                if "attack_success_rate" in metrics:
                    success_rate = metrics["attack_success_rate"]
                    advantage = metrics.get("advantage_over_random", 0)

                    categories = ['Random\n(50%)', f'Observed\n({success_rate:.1%})']
                    values = [0.5, success_rate]
                    colors = ['gray', 'orange' if success_rate > 0.6 else 'green']

                    ax2.bar(categories, values, color=colors, alpha=0.7)
                    ax2.set_ylabel('Attack Success Rate')
                    ax2.set_ylim(0, 1)
                    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
                    ax2.grid(axis='y', alpha=0.3)
                    ax2.set_title('Attack Success Rate', fontsize=10)

            elif "Noise Distribution" in test_name:
                ax2 = plt.subplot(n_tests, 3, idx * 3 + 2)
                if "observed_std" in metrics and "expected_std" in metrics:
                    # Create histogram visualization (placeholder)
                    observed = metrics["observed_std"]
                    expected = metrics["expected_std"]

                    x = ['Expected', 'Observed']
                    y = [expected, observed]
                    colors = ['steelblue', 'green']

                    ax2.bar(x, y, color=colors, alpha=0.7)
                    ax2.set_ylabel('Standard Deviation')
                    ax2.set_title('Noise Scale Comparison', fontsize=10)
                    ax2.grid(axis='y', alpha=0.3)

            elif "Neighboring Database" in test_name:
                ax2 = plt.subplot(n_tests, 3, idx * 3 + 2)
                if "mean_difference" in metrics:
                    mean_diff = metrics["mean_difference"]
                    max_diff = metrics["max_difference"]
                    identical_pct = metrics.get("identical_outputs_pct", 0)

                    x = ['Mean\nDifference', 'Max\nDifference', 'Identical\nOutputs (%)']
                    y = [mean_diff, max_diff, identical_pct / 100]
                    colors = ['steelblue', 'orange', 'green']

                    ax2.bar(x, y, color=colors, alpha=0.7)
                    ax2.set_ylabel('Value')
                    ax2.set_title('Output Differences', fontsize=10)
                    ax2.grid(axis='y', alpha=0.3)

            # Privacy parameters
            ax3 = plt.subplot(n_tests, 3, idx * 3 + 3)
            privacy_text = f"Privacy Budget:\nε = {result['epsilon']}\nδ = {result['delta']:.2e}"
            ax3.text(0.5, 0.5, privacy_text, ha='center', va='center',
                    fontsize=11, family='monospace', transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            ax3.axis('off')

        plt.tight_layout()

        if output_file is None:
            output_file = self.results_dir / "privacy_verification_detailed.png"
        else:
            output_file = Path(output_file)

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Detailed report saved to: {output_file}")

        return fig

    def generate_all_visualizations(self):
        """Generate all visualization reports."""

        print("\n" + "="*70)
        print("GENERATING PRIVACY VERIFICATION VISUALIZATIONS")
        print("="*70 + "\n")

        try:
            self.generate_summary_plot()
            self.generate_detailed_report()

            print("\n" + "="*70)
            print("✓ ALL VISUALIZATIONS GENERATED")
            print("="*70)
            print(f"\nOutput directory: {self.results_dir}/")
            print("  - privacy_verification_summary.png")
            print("  - privacy_verification_detailed.png")
            print("\n")

        except Exception as e:
            print(f"\n✗ Error generating visualizations: {e}")


def main():
    """Generate visualizations from existing results."""

    try:
        visualizer = PrivacyVisualizationReport()
        visualizer.generate_all_visualizations()

        if MATPLOTLIB_AVAILABLE:
            print("💡 TIP: View the generated PNG files to see the results.")
        else:
            print("⚠️  Install matplotlib to enable visualizations:")
            print("    pip install matplotlib")

    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        print("\nPlease run example_privacy_verification.py first to generate test results.")


if __name__ == "__main__":
    main()
