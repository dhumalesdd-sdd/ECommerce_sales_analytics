import csv
import random
import os
from datetime import datetime, timedelta

def generate_dataset():
    os.makedirs('data', exist_ok=True)
    random.seed(42)
    
    cities = ['New York', 'San Francisco', 'Chicago', 'Houston', 'Miami', 'Seattle', 'Austin', 'Boston', 'Denver', 'Atlanta']
    genders = ['Male', 'Female', 'Non-Binary', 'Prefer not to say']
    categories = {
        'Electronics': [
            ('Wireless Noise-Canceling Headphones', 199.99),
            ('4K Ultra HD Smart TV', 649.99),
            ('Bluetooth Portable Speaker', 49.99),
            ('Mechanical Gaming Keyboard', 119.50),
            ('Smart Fitness Watch', 149.00),
            ('Fast Charging Power Bank 20000mAh', 39.99),
            ('Ergonomic Wireless Mouse', 29.99),
        ],
        'Clothing': [
            ('Classic Denim Jacket', 79.99),
            ('Organic Cotton Hoodie', 54.50),
            ('Slim Fit Chinos', 45.00),
            ('Performance Running Shorts', 32.00),
            ('Thermal Winter Parka', 149.95),
            ('Breathable Workout Tee', 24.99),
        ],
        'Home & Kitchen': [
            ('Stainless Steel Cookware Set (10-pc)', 219.99),
            ('Cold Brew Coffee Maker', 34.50),
            ('Smart Air Purifier HEPA', 129.00),
            ('Memory Foam Ergonomic Pillow', 44.99),
            ('Cast Iron Skillet 12-inch', 39.95),
            ('Touchless Electric Kettle', 59.99),
        ],
        'Beauty & Personal Care': [
            ('Hydrating Hyaluronic Acid Serum', 28.00),
            ('Ionic Hair Dryer & Styler', 89.99),
            ('Natural Botanical Body Wash', 18.50),
            ('Mineral Sunscreen SPF 50', 22.00),
            ('Organic Argan Hair Mask', 26.50),
        ],
        'Sports & Outdoors': [
            ('Carbon Fiber Trekking Poles', 58.00),
            ('Inflatable Stand-Up Paddleboard', 349.99),
            ('Non-Slip Eco Yoga Mat', 38.00),
            ('Insulated Hydro Water Bottle 32oz', 29.95),
            ('Lightweight 2-Person Camping Tent', 159.00),
        ],
        'Books & Media': [
            ('Data Science and AI Handbook', 49.99),
            ('Clean Code & Architecture Guide', 44.50),
            ('The Psychology of Decision Making', 21.00),
            ('Atomic Habits and Productivity', 18.99),
            ('Modern Graphic Novel Anthology', 27.50),
        ]
    }
    
    payment_methods = ['Credit Card', 'PayPal', 'Debit Card', 'UPI / NetBanking', 'Apple Pay', 'Cash on Delivery']
    delivery_statuses = ['Delivered', 'Delivered', 'Delivered', 'Shipped', 'Processing', 'Cancelled']
    
    start_date = datetime(2024, 1, 1)
    
    rows = []
    num_records = 1500
    
    for i in range(1, num_records + 1):
        order_id = f"ORD-{10000 + i}"
        customer_id = f"CUST-{random.randint(1001, 1400)}"
        date = start_date + timedelta(days=random.randint(0, 364), hours=random.randint(8, 22), minutes=random.randint(0, 59))
        date_str = date.strftime("%Y-%m-%d")
        
        category = random.choices(
            list(categories.keys()),
            weights=[30, 22, 18, 12, 11, 7],
            k=1
        )[0]
        
        product_name, base_price = random.choice(categories[category])
        quantity = random.choices([1, 2, 3, 4, 5], weights=[55, 25, 12, 5, 3])[0]
        
        # Add slight price fluctuation
        unit_price = round(base_price * random.uniform(0.92, 1.08), 2)
        discount_pct = random.choices([0, 5, 10, 15, 20, 25], weights=[45, 20, 15, 10, 7, 3])[0]
        total_amount = round(quantity * unit_price * (1 - discount_pct / 100.0), 2)
        
        gender = random.choices(genders, weights=[48, 46, 4, 2])[0]
        age = random.choices(
            [random.randint(18, 25), random.randint(26, 35), random.randint(36, 50), random.randint(51, 70)],
            weights=[25, 40, 23, 12]
        )[0]
        
        city = random.choices(cities, weights=[18, 16, 12, 10, 9, 8, 8, 7, 6, 6])[0]
        payment_method = random.choices(payment_methods, weights=[38, 22, 15, 12, 9, 4])[0]
        delivery_status = random.choice(delivery_statuses)
        
        # Rating roughly correlated with delivery status and price satisfaction
        if delivery_status == 'Cancelled':
            rating = round(random.uniform(1.0, 2.5), 1)
        else:
            rating = round(random.choices([5.0, 4.5, 4.0, 3.5, 3.0, 2.0, 1.0], weights=[40, 25, 18, 8, 5, 2, 2])[0], 1)
            
        rows.append([
            order_id, date_str, customer_id, gender, age, city,
            category, product_name, quantity, unit_price, discount_pct,
            total_amount, payment_method, rating, delivery_status
        ])
        
    # Introduce deliberate duplicates (25 duplicate rows)
    duplicates = [rows[random.randint(0, len(rows)-1)].copy() for _ in range(25)]
    rows.extend(duplicates)
    random.shuffle(rows)
    
    # Introduce deliberate missing values (around 35 nulls across some columns)
    for _ in range(35):
        r_idx = random.randint(0, len(rows) - 1)
        col_to_null = random.choice([4, 10, 13]) # Age, Discount_pct, Rating
        rows[r_idx][col_to_null] = ""
        
    for _ in range(15):
        r_idx = random.randint(0, len(rows) - 1)
        rows[r_idx][5] = "" # City missing
        
    headers = [
        "Order_ID", "Date", "Customer_ID", "Gender", "Age", "City",
        "Product_Category", "Product_Name", "Quantity", "Unit_Price",
        "Discount_Percent", "Total_Amount", "Payment_Method", "Customer_Rating", "Delivery_Status"
    ]
    
    target_path = os.path.join("data", "ecommerce_sales_data.csv")
    with open(target_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    print(f"Generated {len(rows)} records in {target_path}")

if __name__ == "__main__":
    generate_dataset()
