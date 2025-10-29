import unittest
import os
import sys
import tempfile
import shutil
from app import app, generate_data, train_models, predict_tumor

class TumorClassificationTestCase(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = app.test_client()
        self.app.testing = True

        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.original_db = 'data/tumors.db'

        # Override database path for testing
        app.config['DATABASE'] = os.path.join(self.test_dir, 'test_tumors.db')

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up test database
        if os.path.exists(app.config['DATABASE']):
            os.remove(app.config['DATABASE'])
        os.rmdir(self.test_dir)

    def test_home_page(self):
        """Test that home page loads successfully."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Tumor Classification System', response.data)

    def test_dashboard_page(self):
        """Test that dashboard page loads successfully."""
        response = self.app.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Dashboard', response.data)

    def test_predict_page(self):
        """Test that predict page loads successfully."""
        response = self.app.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Make Tumor Prediction', response.data)

    def test_models_page_without_training(self):
        """Test models page when models are not trained."""
        response = self.app.get('/models')
        self.assertEqual(response.status_code, 302)  # Should redirect to dashboard

    def test_upload_page(self):
        """Test that upload page loads successfully."""
        response = self.app.get('/upload')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Upload Dataset', response.data)

    def test_data_generation(self):
        """Test that data generation works correctly."""
        df = generate_data(100)
        self.assertEqual(len(df), 100)
        self.assertIn('diagnosis', df.columns)
        self.assertIn('age', df.columns)
        self.assertIn('tumor_size', df.columns)
        self.assertTrue(all(df['diagnosis'].isin(['benign', 'malignant'])))

    def test_model_training(self):
        """Test that model training works correctly."""
        # Generate test data
        df = generate_data(200)
        conn = app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + app.config['DATABASE']
        df.to_sql('tumors', conn, index=False, if_exists='replace')

        # Test model training
        results = train_models()
        self.assertIn('svm_accuracy', results)
        self.assertIn('rf_accuracy', results)
        self.assertTrue(0 <= results['svm_accuracy'] <= 1)
        self.assertTrue(0 <= results['rf_accuracy'] <= 1)

    def test_prediction_functionality(self):
        """Test prediction functionality."""
        # Train models first
        df = generate_data(200)
        conn = app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + app.config['DATABASE']
        df.to_sql('tumors', conn, index=False, if_exists='replace')

        train_models()

        # Test prediction
        features = [45, 2.5, 65.2, 0.85, 0.72]
        prediction = predict_tumor(features)

        self.assertIsNotNone(prediction)
        self.assertIn('svm_prediction', prediction)
        self.assertIn('rf_prediction', prediction)
        self.assertIn('svm_class', prediction)
        self.assertIn('rf_class', prediction)
        self.assertTrue(prediction['svm_class'] in ['benign', 'malignant'])
        self.assertTrue(prediction['rf_class'] in ['benign', 'malignant'])

if __name__ == '__main__':
    unittest.main()
