import unittest
import json
import sys
sys.path.insert(0, '.')

from metadata import appmetadata


class TestBatchSizeConfig(unittest.TestCase):
    """Test that the tpModelBatchSize parameter is properly defined in metadata."""
    
    def test_metadata_includes_batch_size_parameter(self):
        """Verify that the metadata includes the tpModelBatchSize parameter."""
        metadata = appmetadata()
        
        # Get the JSON representation
        metadata_dict = json.loads(metadata.jsonify())
        
        # Find the tpModelBatchSize parameter
        batch_size_param = None
        for param in metadata_dict['parameters']:
            if param['name'] == 'tpModelBatchSize':
                batch_size_param = param
                break
        
        self.assertIsNotNone(batch_size_param, "tpModelBatchSize parameter should exist in metadata")
        self.assertEqual(batch_size_param['type'], 'integer', "tpModelBatchSize should be an integer")
        self.assertEqual(batch_size_param['default'], 200, "Default batch size should be 200")
        self.assertIn('VRAM', batch_size_param['description'], 
                     "Description should mention VRAM usage")


if __name__ == '__main__':
    unittest.main()
