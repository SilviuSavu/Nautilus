import React from 'react';
import { Modal, Form, InputNumber, Button, Space, Alert } from 'antd';
import { EditOutlined } from '@ant-design/icons';
import { IBOrder, OrderModificationData } from '../hooks/useOrderManagement';

interface OrderModificationModalProps {
  visible: boolean;
  order: IBOrder | null;
  loading?: boolean;
  onCancel: () => void;
  onModify: (orderId: string, values: OrderModificationData) => Promise<{ success: boolean }>;
}

export const OrderModificationModal: React.FC<OrderModificationModalProps> = ({
  visible,
  order,
  loading = false,
  onCancel,
  onModify
}) => {
  const [form] = Form.useForm();

  const handleSubmit = async (values: OrderModificationData) => {
    if (!order) return;
    
    const result = await onModify(order.order_id, values);
    if (result.success) {
      onCancel();
      form.resetFields();
    }
  };

  const handleCancel = () => {
    onCancel();
    form.resetFields();
  };

  if (!order) return null;

  return (
    <Modal
      title="Modify Order"
      open={visible}
      onCancel={handleCancel}
      footer={null}
      width={500}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={{
          quantity: order.total_quantity,
          limit_price: order.limit_price,
          stop_price: order.stop_price,
        }}
      >
        <Alert
          message={`Modifying order for ${order.symbol}`}
          description={`Order ID: ${order.order_id} | Current Status: ${order.status}`}
          type="info"
          style={{ marginBottom: '16px' }}
        />

        <Form.Item
          label="Quantity"
          name="quantity"
          rules={[
            { required: true, message: 'Please enter quantity' },
            { type: 'number', min: 1, message: 'Quantity must be at least 1' }
          ]}
        >
          <InputNumber
            min={1}
            style={{ width: '100%' }}
            placeholder="Enter new quantity"
          />
        </Form.Item>

        {order.order_type !== 'MKT' && (
          <Form.Item
            label="Limit Price"
            name="limit_price"
            rules={[
              { type: 'number', min: 0.01, message: 'Price must be greater than 0' }
            ]}
          >
            <InputNumber
              min={0.01}
              step={0.01}
              precision={2}
              style={{ width: '100%' }}
              placeholder="Enter new limit price"
              addonBefore="$"
            />
          </Form.Item>
        )}

        {(order.order_type === 'STP' || order.order_type === 'STP_LMT') && (
          <Form.Item
            label="Stop Price"
            name="stop_price"
            rules={[
              { type: 'number', min: 0.01, message: 'Price must be greater than 0' }
            ]}
          >
            <InputNumber
              min={0.01}
              step={0.01}
              precision={2}
              style={{ width: '100%' }}
              placeholder="Enter new stop price"
              addonBefore="$"
            />
          </Form.Item>
        )}

        <Form.Item style={{ marginBottom: 0 }}>
          <Space>
            <Button
              type="primary"
              htmlType="submit"
              icon={<EditOutlined />}
              loading={loading}
            >
              Modify Order
            </Button>
            <Button onClick={handleCancel}>
              Cancel
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </Modal>
  );
};