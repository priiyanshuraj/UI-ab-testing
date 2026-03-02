import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the page
st.set_page_config(page_title="A/B Test Analyzer", layout="wide")

st.title("E-commerce UI A/B Testing Analysis")
st.markdown("Upload your A/B test data and countries data to dynamically analyze the results through probability, A/B testing (simulations & z-test), and logistic regression.")

# --- 1. INTERACTIVE DATA UPLOADS ---
st.sidebar.header("1. Upload Data")
ab_data_file = st.sidebar.file_uploader("Upload A/B Test Data (CSV)", type=["csv"])
countries_file = st.sidebar.file_uploader("Upload Countries Data (CSV)", type=["csv"])

if ab_data_file is not None and countries_file is not None:
    # Load data
    df = pd.read_csv(ab_data_file)
    countries = pd.read_csv(countries_file)
    
    with st.expander("Preview Raw Data"):
        st.write("A/B Data:", df.head())
        st.write("Countries Data:", countries.head())
    
    # Data Cleaning 
    df_clean = df.drop(df[(df['group'] == 'treatment') & (df['landing_page'] == 'old_page')].index)
    df_clean = df_clean.drop(df_clean[(df_clean['group'] == 'control') & (df_clean['landing_page'] == 'new_page')].index)
    
    # Drop duplicates
    if df_clean.user_id.duplicated().sum() > 0:
        dup_index = df_clean[df_clean.user_id.duplicated()].index
        df_clean = df_clean.drop(dup_index)
        
    st.success(f"Data cleaned successfully! Rows ready for analysis: {df_clean.shape[0]}")
    # --- NEW: PART I: BASIC PROBABILITY ---
    st.header("Basic Probability & Observed Rates")
    st.markdown("Before running formal hypothesis tests, let's look at the basic conversion probabilities in our dataset.")
    
    # Calculate probabilities
    overall_conv = df_clean['converted'].mean()
    control_conv = df_clean.query('group == "control"')['converted'].mean()
    treatment_conv = df_clean.query('group == "treatment"')['converted'].mean()
    prob_new_page = (df_clean['landing_page'] == "new_page").mean()
    
    # Display metrics neatly
    prob_col1, prob_col2, prob_col3, prob_col4 = st.columns(4)
    prob_col1.metric("Overall Conversion", f"{overall_conv:.4%}")
    prob_col2.metric("Control Conversion", f"{control_conv:.4%}")
    prob_col3.metric("Treatment Conversion", f"{treatment_conv:.4%}")
    prob_col4.metric("Prob. of Receiving New Page", f"{prob_new_page:.4%}")
    
    # Show the observed difference dynamically
    obs_diff = treatment_conv - control_conv
    
    if obs_diff > 0:
        direction = "higher"
        display_diff = obs_diff
    elif obs_diff < 0:
        direction = "lower"
        display_diff = abs(obs_diff) # Removes the negative sign since we use the word "lower"
    else:
        direction = "the exact same"
        display_diff = 0

    if obs_diff != 0:
        st.info(f"**Observed Difference:**\n\n The treatment group (new page) has a conversion rate that is **{display_diff:.4%} {direction}** than the control group (old page).\n\n Next, we will use A/B testing to figure out if this tiny difference is statistically significant or just due to random chance.")
    else:
        st.info(f"**Observed Difference:**\n\n The treatment group (new page) has a conversion rate that is **{direction}** as the control group (old page).")
        
    st.divider() # Adds a nice visual line to separate sectionsdds a nice visual line to separate sections

    # --- 2 & 3. DYNAMIC SIMULATIONS & ADJUSTABLE CONFIDENCE INTERVALS ---
    st.sidebar.header("2. Simulation Parameters")
    num_simulations = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=20000, value=10000, step=1000)
    alpha_level = st.sidebar.slider("Alpha Level (Significance)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

    st.header("A/B Test")
    
    # Calculate probabilities and counts
    p_new = df_clean.converted.mean()
    p_old = df_clean.converted.mean()
    n_new = df_clean.query('landing_page == "new_page"').shape[0]
    n_old = df_clean.query('landing_page == "old_page"').shape[0]
    
    pdiff_actual = df_clean.query('group == "treatment"').converted.mean() - df_clean.query('group == "control"').converted.mean()

    with st.spinner(f"Running {num_simulations} simulations..."):
        # Binomial stimulation
        new_page_converted = np.random.binomial(n_new, p_new, num_simulations) / n_new
        old_page_converted = np.random.binomial(n_old, p_old, num_simulations) / n_old
        p_diffs = new_page_converted - old_page_converted
        p_diffs_mean = p_diffs.mean()
        
        # Compute p-value
        null_vals = np.random.normal(0, np.std(p_diffs), num_simulations)
        p_value = (null_vals > pdiff_actual).mean()
        
        # Calculate Z-score using statsmodels
        convert_old = df_clean.query('group == "control" & converted == 1').user_id.count()
        convert_new = df_clean.query('group == "treatment" & converted == 1').user_id.count()
        z_score, p_val_z = sm.stats.proportions_ztest([convert_new, convert_old], [n_new, n_old], alternative='larger')

    # --- 4. LIVE VISUALIZATIONS ---
    st.subheader("Distribution of Differences Under Null Hypothesis")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(p_diffs, bins=50, ax=ax, color='skyblue', kde=False)
    
    # Confidence Interval Bounds
    low_bound = np.percentile(p_diffs, (alpha_level/2) * 100)
    upper_bound = np.percentile(p_diffs, (1 - alpha_level/2) * 100)
    
    ax.axvline(x=pdiff_actual, color='red', linestyle='--', linewidth=2, label=f'Actual Difference ({pdiff_actual:.4f})')
    ax.axvline(x=p_diffs_mean, color='gray', linestyle='-', linewidth=2, label=f'Null Mean ({p_diffs_mean:.4f})')
    ax.axvline(x=low_bound, color='orange', linestyle=':', linewidth=2, label=f'CI Lower Bound')
    ax.axvline(x=upper_bound, color='orange', linestyle=':', linewidth=2, label=f'CI Upper Bound')
    
    ax.set_xlabel('Difference in Conversion Rates (p_new - p_old)')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

    # --- 5. AUTOMATED RESULTS INTERPRETATION ---
    st.subheader("Statistical Results & Interpretation")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Simulated P-Value", f"{p_value:.4f}")
    col2.metric("Z-Test P-Value", f"{p_val_z:.4f}")
    col3.metric("Z-Score", f"{z_score:.4f}")
    
    st.markdown("#### 📊 Statistical Conclusion")
    if p_value < alpha_level:
        st.success(f"The p-value ({p_value:.4f}) is less than the alpha level ({alpha_level}). We **reject the null hypothesis**. There is sufficient evidence to suggest the new page leads to more conversions.")
        
        st.markdown("#### 🎯 Business Recommendation")
        st.success(
            "🎉 **Launch the New Page!**\n\n"
            "Based on our analysis, the new page brings in noticeably more conversions than the old page. "
            "We are highly confident that this improvement is real and not just a fluke. Switching to the new design is a solid business decision."
        )
    else:
        st.error(f"The p-value ({p_value:.4f}) is greater than the alpha level ({alpha_level}). We **fail to reject the null hypothesis**. There is not enough evidence to suggest the new page leads to more conversions.")
        
        st.markdown("#### 🎯 Business Recommendation")
        st.warning(
            "🛑 **Keep the Old Page.**\n\n"
            "The data shows that the new page did **not** perform significantly better than the old page. "
            "Any minor differences we saw were likely just random chance. It is safer to stick with the current design, "
            "save deployment costs, and perhaps try testing a completely different idea next time."
        )

    st.divider()

    # --- 6. LOGISTIC REGRESSION TOGGLES ---
    st.header("Analysis using Logistic Regression Model")
    st.markdown("Use the toggles below to configure the logistic regression model and view the impact on statistical significance.")
    
    # Prepare base data for regression
    df_reg = df_clean.copy()
    df_reg['intercept'] = 1
    
    # Ensure 'converted' is an integer
    df_reg['converted'] = df_reg['converted'].astype(int)
    
    # Convert group to dummy variable and FORCE it to be an integer (1/0) instead of boolean (True/False)
    df_reg['ab_page'] = pd.get_dummies(df_reg['group'])['treatment'].astype(int)
    
    # Merge with countries
    df_reg = df_reg.set_index('user_id').join(countries.set_index('user_id'))
    
    # Create dummies for countries and FORCE them to be integers
    country_dummies = pd.get_dummies(df_reg['country']).astype(int)
    df_reg = df_reg.join(country_dummies)
    
    # Default features
    features = ['intercept', 'ab_page']
    
    col_a, col_b = st.columns(2)
    with col_a:
        add_countries = st.checkbox("Add Country Variables (US, UK)")
    with col_b:
        add_interactions = st.checkbox("Add Interaction Terms (Country * ab_page)")
        
    if add_countries:
        # Assuming CA is baseline to prevent multicollinearity
        if 'US' in df_reg.columns and 'UK' in df_reg.columns:
            features.extend(['US', 'UK'])
            
    if add_interactions and add_countries:
        df_reg['US_ab_page'] = df_reg['US'] * df_reg['ab_page']
        df_reg['UK_ab_page'] = df_reg['UK'] * df_reg['ab_page']
        features.extend(['US_ab_page', 'UK_ab_page'])
    elif add_interactions and not add_countries:
        st.warning("⚠️ Please enable 'Country Variables' first to add interaction terms.")
        
    # Run Logistic Regression
    try:
        logit_mod = sm.Logit(df_reg['converted'], df_reg[features])
        results = logit_mod.fit(disp=0)
        
        # --- NEW CODE FOR CLEAN UI ---
        # Extract the results into a clean Pandas DataFrame
        summary_df = pd.DataFrame({
            'Coefficient': results.params,
            'Std Error': results.bse,
            'z-value': results.tvalues, # tvalues act as z-values in statsmodels Logit
            'P-value': results.pvalues,
            'CI Lower': results.conf_int()[0],
            'CI Upper': results.conf_int()[1]
        })
        
        st.write("### Logistic Regression Coefficients")
        # Display as a nicely formatted Streamlit dataframe
        st.dataframe(summary_df.style.format({
            'Coefficient': '{:.4f}',
            'Std Error': '{:.4f}',
            'z-value': '{:.4f}',
            'P-value': '{:.4f}',
            'CI Lower': '{:.4f}',
            'CI Upper': '{:.4f}'
        }).highlight_min(subset=['P-value'], color='lightgreen'), use_container_width=True)
        # -----------------------------
        
        # 
        # Quick automated interpretation for regression
        pvals = results.pvalues
        significant_vars = pvals[pvals < alpha_level].index.tolist()
        significant_vars = [v for v in significant_vars if v != 'intercept']
        
        st.markdown("#### 📊 Statistical Conclusion")
        if significant_vars:
            st.info(f"Statistically significant variables (p < {alpha_level}): **{', '.join(significant_vars)}**")
            
            st.markdown("#### 💡 Business Insight")
            clean_vars = [v.replace('_', ' ') for v in significant_vars]
            st.success(
                f"**What else matters?**\n\n"
                "Our deep dive shows that factors like **{', '.join(clean_vars)}** "
                f"actually have a real impact on whether a user decides to convert or not."
            )
        else:
            st.warning(f"No variables (other than intercept) are statistically significant at an alpha level of {alpha_level}.")
            
            st.markdown("#### 💡 Business Insight")
            st.info(
                "**Do other factors matter?**\n\n"
                "We looked to see if extra factors (like the user's country or interaction terms)"
                "changed how people reacted to the pages. The data tells us that these extra factors **did not** "
                "have any meaningful impact on our conversion rates. The users' behavior is fairly consistent across the board."
            )
           
    except Exception as e:
        st.error(f"Error fitting the logistic regression model: {e}")
else:
    st.info("👈 Please upload both `ab_data.csv` and `countries.csv` in the sidebar to view the interactive analysis.")
# do streamlit run app.py to get the app