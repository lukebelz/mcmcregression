from flask import Flask, render_template, send_file, request, jsonify
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pickle

app = Flask(__name__)

data_vars = {"independent_vars": ['Wood Deck SF', 'Open Porch SF', 'Total Bsmt SF', 'Yr Sold', 'Mo Sold'], "target_var": ["SalePrice"], "file_name": "AmesHousing.csv"}

# Load the saved pickle file
with open('model_results.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract the loaded data
results_mcmc = data['results_mcmc']
results_ols = data['results_ols']

def get_min_max_sse(results_ols, results_mcmc):
    all_sse_values = []

    # Collect SSE values from OLS results
    for combination in results_ols:
        all_sse_values.append(results_ols[combination]['SSE'])
    
    # Collect SSE values from MCMC results
    for combination in results_mcmc:
        all_sse_values.append(results_mcmc[combination]['SSE'])
    
    # Compute min and max SSE
    min_sse = min(all_sse_values)
    max_sse = max(all_sse_values)
    
    return min_sse, max_sse

def min_max_normalize(value, min_value, max_value):
    if max_value == min_value:
        # Avoid division by zero, return 0 in case of constant values
        return 0
    return (value - min_value) / (max_value - min_value)

@app.route('/')
def index():
    independent_vars = data_vars["independent_vars"]
    variable_names = [name.replace("_", " ").title() for name in independent_vars]
    return render_template('index.html', independent_vars=independent_vars, variable_names=variable_names, indicies=range(len(independent_vars)))

@app.route('/plot', methods=['POST'])
def plot_graph():
    # Get selected categories from POST request
    selected_categories = request.json['selectedCategories']
    
    print(selected_categories)
    # Create a tuple from selected categories
    category_tuple = tuple(sorted(tuple(selected_categories)))

    if category_tuple in results_ols and category_tuple in results_mcmc:
        # Data
        categories = ['Normalized\nSSE', 'R_squared', 'R_squared\nAdjusted', 'P-Value']
        bar_width = 0.4
        y_pos = range(len(categories))

        # Create a bigger figure
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the figure size to make the graph bigger

        # Fetch data for the selected model
        mlr_model = [v for k, v in list(results_ols[category_tuple].items()) if k != 'trace']
        mcmc_model = [v for k, v in list(results_mcmc[category_tuple].items()) if k != 'trace']

        # Normalize SSEs
        sse_min, sse_max = get_min_max_sse(results_ols, results_mcmc)
        mlr_model[0] = min_max_normalize(mlr_model[0], sse_min, sse_max)
        mcmc_model[0] = min_max_normalize(mcmc_model[0], sse_min, sse_max)

        # Determine the x-axis limit for consistent scaling
        max_value = max(max(mlr_model), max(mcmc_model))
        ax.set_xlim(0, 1)  # Leave some padding on the right

        # Plot bars with categories on the right side
        ax.barh(y_pos, mlr_model, height=bar_width, label='MLR', color='#e74c3c', align='center')  # Match button red
        ax.barh([y + bar_width for y in y_pos], mcmc_model, height=bar_width, label='MCMC', color='#f39c12', align='center')  # Match button orange

        # Display the true values fixed to the left side of the graph, just outside the spine
        for i, v in enumerate(mlr_model):
            ax.text(-0.05 * max_value, i, f'{v:.3f}', va='center', ha='right', fontsize=12, color='black')  # Adjust position outside the spine
        for i, v in enumerate(mcmc_model):
            ax.text(-0.05 * max_value, i + bar_width, f'{v:.3f}', va='center', ha='right', fontsize=12, color='black')  # Adjust position outside the spine

        # Customize ticks, making categories appear on the right
        ax.set_yticks([y + bar_width / 2 for y in y_pos])
        ax.set_yticklabels(categories, fontsize=14)
        ax.tick_params(axis='y', labelright=True, labelleft=False)  # Categories on the right

        # Custom background color to match HTML background
        fig.patch.set_facecolor('#f4f4f4')
        ax.set_facecolor('#f4f4f4')

        # Add title and legend
        ax.set_title('MLR vs MCMC', fontsize=18, fontweight='bold')
        ax.legend()

        # Add a grid for better readability
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        # Remove spines on the top and right for a cleaner look
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Save the plot to a bytes buffer
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        plt.close(fig)

        # Return the image
        output.seek(0)
        return send_file(output, mimetype='image/png')
    else:
        print(f'{category_tuple} Not found')
        return jsonify({'error': 'No data found for the selected categories.'}), 400

if __name__ == '__main__':
    app.run(debug=True)