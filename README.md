# Budget Manager  

## Overview  
**Budget Manager** is a collaborative project developed during an IBM summer training program. It leverages **machine learning** to help users make **informed financial decisions** by analyzing spending patterns and providing insights into potential savings opportunities.  

By comparing user-inputted expenses with similar transactions in the same category, Budget Manager empowers users to:  
- Identify areas where they can optimize spending.  
- Track their budget more effectively.  
- Make smarter, financially responsible choices.  

## Features  
- **Machine Learning-Powered Insights**: Uses a **Random Forest Classifier** to predict expenses based on historical transaction data.  
- **Category-wise Expenditure Analysis**: Provides average spending benchmarks for different expense categories.  
- **Budget Tracking & Optimization**: Identifies expenses exceeding predefined limits and suggests savings.  
- **User-Friendly Interface**: Built with **Streamlit**, allowing users to log transactions effortlessly.  
- **Detailed Transaction Logging**: Users can input:  
  - **Date, Category, Subcategory**  
  - **Remarks, Amount, Income/Expense Type**  
  - **Currency (INR)**  

## Technologies Used  
- **Machine Learning:** Random Forest Classifier  
- **Frontend:** Streamlit (Python-based UI framework)  
- **Backend & Data Processing:** Python, Pandas  
- **Database:** CSV-based storage (expandable to SQL-based solutions)  

## Installation  

### Prerequisites  
- Python 3.7 or higher  
- Pip (Python package manager)  

### Steps  
1. **Clone the Repository:**  
   ```bash
   git clone <repository-url>
   cd Budget-Manager
2. **Create a Virtual Environment (Optional, Recommended):**
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
4. **Run the Application:**
   ```bash
    streamlit run app.py
5. **Access the Web App:**
    Open your browser and visit http://localhost:8501 to start using the Budget Manager.

## Usage
- **Log Your Transactions:** Enter details such as date, category, subcategory, and amount.
- **Analyze Spending Trends:** View average spending insights for different categories.
- **Receive Budget Optimization Suggestions:** Get alerts when expenses exceed recommended limits.
- **Improve Financial Planning:** Utilize AI-driven insights to refine your budgeting strategies.

## Future Enhancements
- Integration with Bank APIs for automatic expense tracking.
- Advanced Predictive Analysis for better financial forecasting.
- Multi-Currency Support to manage finances globally.

## Contributions
Special thanks to Shriya Gupta and Tushansh Bajaj for their instrumental contributions to this group project.

## License
This project is licensed under the MIT License.

## Acknowledgments
We are excited to see how this Budget Manager improves financial planning and transforms expense management!
