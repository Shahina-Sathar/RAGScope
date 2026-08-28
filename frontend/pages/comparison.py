"""Comparison page - Visual comparison of experiments."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.api_client import RAGScopeAPI


def render_comparison(api_client: RAGScopeAPI):
    """Render comparison page."""
    st.markdown('<p class="main-header">📊 Experiment Comparison</p>', unsafe_allow_html=True)
    st.markdown("Compare multiple experiments side-by-side with visual analytics")
    st.markdown("---")
    
    # Check API connection
    if not api_client.health_check():
        st.error("❌ Backend API is not connected")
        return
    
    # Load completed experiments
    try:
        experiments = api_client.list_experiments()
        completed_experiments = [e for e in experiments if e.get("status") == "completed"]
        
        if len(completed_experiments) < 2:
            st.warning("⚠️ Need at least 2 completed experiments to compare")
            st.info(f"Currently have {len(completed_experiments)} completed experiment(s)")
            return
        
        # Experiment selection
        st.subheader("🔧 Select Experiments")
        
        experiment_options = {
            f"{e.get('name')} ({e.get('id')[:8]}...)": e.get("id")
            for e in completed_experiments
        }
        
        selected_names = st.multiselect(
            "Choose experiments to compare (minimum 2)",
            options=list(experiment_options.keys()),
            default=list(experiment_options.keys())[:2] if len(experiment_options) >= 2 else [],
            help="Select 2 or more experiments to compare"
        )
        
        if len(selected_names) < 2:
            st.warning("Please select at least 2 experiments")
            return
        
        selected_ids = [experiment_options[name] for name in selected_names]
        
        if st.button("📊 Compare Experiments", type="primary"):
            with st.spinner("Comparing experiments..."):
                try:
                    comparison = api_client.compare_experiments(selected_ids)
                    st.session_state["comparison_result"] = comparison
                except Exception as e:
                    st.error(f"❌ Error comparing experiments: {e}")
                    return
        
        # Display comparison if available
        if "comparison_result" in st.session_state:
            comparison = st.session_state["comparison_result"]
            
            st.markdown("---")
            st.subheader("📈 Metrics Comparison")
            
            # Metrics overview table
            experiments_data = comparison.get("experiments", [])
            metrics_comparison = comparison.get("metrics_comparison", {})
            
            if experiments_data:
                # Create metrics table
                metrics_data = []
                for exp in experiments_data:
                    metrics_data.append({
                        "Experiment": exp.get("name", "Unknown"),
                        "Avg Latency (ms)": exp.get("avg_latency_ms", 0),
                        "Avg Faithfulness": exp.get("avg_faithfulness", 0),
                        "Avg Relevancy": exp.get("avg_relevancy", 0),
                        "Avg Precision": exp.get("avg_precision", 0),
                        "Avg Recall": exp.get("avg_recall", 0),
                        "Avg Grounding": exp.get("avg_grounding_score", 0),
                    })
                
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, use_container_width=True)
                
                # Visualization tabs
                viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Bar Charts", "📉 Radar Chart", "📋 Detailed View"])
                
                with viz_tab1:
                    st.markdown("### Metric Comparison (Bar Charts)")
                    
                    # Select metric to visualize
                    metric_options = [
                        ("avg_latency_ms", "Latency (ms)", False),  # Lower is better
                        ("avg_faithfulness", "Faithfulness", True),
                        ("avg_relevancy", "Relevancy", True),
                        ("avg_precision", "Precision", True),
                        ("avg_recall", "Recall", True),
                        ("avg_grounding_score", "Grounding", True),
                    ]
                    
                    selected_metric = st.selectbox(
                        "Select metric to visualize",
                        options=[m[1] for m in metric_options],
                        index=1
                    )
                    
                    # Find the metric key
                    metric_key = next(m[0] for m in metric_options if m[1] == selected_metric)
                    higher_is_better = next(m[2] for m in metric_options if m[1] == selected_metric)
                    
                    # Create bar chart
                    df_chart = pd.DataFrame({
                        "Experiment": [exp.get("name", "Unknown") for exp in experiments_data],
                        "Value": [exp.get(metric_key, 0) or 0 for exp in experiments_data]
                    })
                    
                    fig = px.bar(
                        df_chart,
                        x="Experiment",
                        y="Value",
                        title=f"{selected_metric} Comparison",
                        color="Value",
                        color_continuous_scale="Viridis" if higher_is_better else "Viridis_r",
                    )
                    
                    fig.update_layout(
                        xaxis_title="Experiment",
                        yaxis_title=selected_metric,
                        showlegend=False,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Highlight best
                    if experiments_data:
                        values = [exp.get(metric_key, 0) or 0 for exp in experiments_data]
                        if higher_is_better:
                            best_idx = values.index(max(values))
                        else:
                            best_idx = values.index(min(values))
                        
                        best_exp = experiments_data[best_idx]
                        st.success(
                            f"🏆 **Best for {selected_metric}:** {best_exp.get('name')} "
                            f"({values[best_idx]:.2f})"
                        )
                
                with viz_tab2:
                    st.markdown("### Radar Chart (Multi-Metric View)")
                    
                    # Normalize metrics to 0-1 scale for radar chart
                    radar_metrics = ["avg_faithfulness", "avg_relevancy", "avg_precision", "avg_recall", "avg_grounding_score"]
                    radar_labels = ["Faithfulness", "Relevancy", "Precision", "Recall", "Grounding"]
                    
                    # Create radar chart data
                    radar_data = []
                    for exp in experiments_data:
                        values = [exp.get(metric, 0) or 0 for metric in radar_metrics]
                        radar_data.append({
                            "Experiment": exp.get("name", "Unknown"),
                            **{label: val for label, val in zip(radar_labels, values)}
                        })
                    
                    # Create radar chart using plotly
                    fig = go.Figure()
                    
                    for exp_data in radar_data:
                        exp_name = exp_data["Experiment"]
                        values = [exp_data[label] for label in radar_labels]
                        values.append(values[0])  # Close the radar chart
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=radar_labels + [radar_labels[0]],
                            name=exp_name,
                            fill='toself'
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=True,
                        title="Multi-Metric Radar Chart"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("💡 **Tip:** Larger area = better overall performance across all metrics")
                
                with viz_tab3:
                    st.markdown("### Detailed Metrics View")
                    
                    for exp in experiments_data:
                        with st.expander(f"📋 {exp.get('name')}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Performance Metrics:**")
                                st.metric("Latency", f"{exp.get('avg_latency_ms', 0):.0f} ms")
                                st.metric("Faithfulness", f"{exp.get('avg_faithfulness', 0):.2f}")
                                st.metric("Relevancy", f"{exp.get('avg_relevancy', 0):.2f}")
                            
                            with col2:
                                st.metric("Precision", f"{exp.get('avg_precision', 0):.2f}")
                                st.metric("Recall", f"{exp.get('avg_recall', 0):.2f}")
                                st.metric("Grounding", f"{exp.get('avg_grounding_score', 0):.2f}")
                            
                            st.markdown("**Configuration:**")
                            config = exp.get("config", {})
                            st.json(config)
            
            # Failure comparison
            st.markdown("---")
            st.subheader("⚠️ Failure Pattern Comparison")
            
            failures_comparison = comparison.get("failures_comparison", {})
            
            if failures_comparison:
                # Create failure comparison table
                failure_data = []
                for exp_id, data in failures_comparison.items():
                    exp_name = next((e.get("name") for e in experiments_data if e.get("id") == exp_id), exp_id[:8])
                    failure_data.append({
                        "Experiment": exp_name,
                        "Total Queries": data.get("total_queries", 0),
                        "Failures": data.get("queries_with_failures", 0),
                        "Failure Rate": f"{data.get('failure_rate', 0) * 100:.1f}%",
                    })
                
                df_failures = pd.DataFrame(failure_data)
                st.dataframe(df_failures, use_container_width=True)
                
                # Failure types breakdown
                st.markdown("### Failure Types Breakdown")
                
                all_failure_types = set()
                for data in failures_comparison.values():
                    all_failure_types.update(data.get("failure_counts", {}).keys())
                
                if all_failure_types:
                    failure_type_data = []
                    for exp_id, data in failures_comparison.items():
                        exp_name = next((e.get("name") for e in experiments_data if e.get("id") == exp_id), exp_id[:8])
                        for failure_type in all_failure_types:
                            count = data.get("failure_counts", {}).get(failure_type, 0)
                            failure_type_data.append({
                                "Experiment": exp_name,
                                "Failure Type": failure_type,
                                "Count": count
                            })
                    
                    df_failure_types = pd.DataFrame(failure_type_data)
                    
                    fig = px.bar(
                        df_failure_types,
                        x="Experiment",
                        y="Count",
                        color="Failure Type",
                        barmode="group",
                        title="Failure Types by Experiment"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No failures detected in any experiment!")
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            recommendations = comparison.get("recommendations", [])
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"**{i}.** {rec}")
            else:
                st.info("No specific recommendations at this time.")
            
            # Raw comparison data
            with st.expander("🔧 Raw Comparison Data"):
                st.json(comparison)
    
    except Exception as e:
        st.error(f"❌ Error loading experiments: {e}")
        import traceback
        st.code(traceback.format_exc())
