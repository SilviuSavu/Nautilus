import React, { useState } from 'react';
import { 
  Modal, 
  Form, 
  Input, 
  Select, 
  InputNumber, 
  Button, 
  Space, 
  Alert, 
  Divider,
  Row,
  Col,
  Typography,
  Checkbox 
} from 'antd';
import { 
  ShoppingCartOutlined, 
  DollarOutlined, 
  ClockCircleOutlined 
} from '@ant-design/icons';

const { Option } = Select;
const { Text } = Typography;

// Separate component for order summary to prevent rendering issues
const OrderSummaryText: React.FC<{ form: any; orderType: string }> = ({ form, orderType }) => {
  const action = Form.useWatch('action', form) || 'BUY';
  const quantity = Form.useWatch('quantity', form) || 0;
  const symbol = Form.useWatch('symbol', form) || '[SYMBOL]';
  const limitPrice = Form.useWatch('limit_price', form) || 0;
  const stopPrice = Form.useWatch('stop_price', form) || 0;
  const trailAmount = Form.useWatch('trail_amount', form);
  const trailPercent = Form.useWatch('trail_percent', form) || 0;
  const takeProfitPrice = Form.useWatch('take_profit_price', form) || 0;
  const stopLossPrice = Form.useWatch('stop_loss_price', form) || 0;
  const ocaGroup = Form.useWatch('oca_group', form) || '[GROUP]';
  const outsideRth = Form.useWatch('outside_rth', form);
  const hidden = Form.useWatch('hidden', form);

  let priceText = '';
  switch (orderType) {
    case 'MKT':
      priceText = ' at market price';
      break;
    case 'LMT':
      priceText = ` at limit $${limitPrice}`;
      break;
    case 'STP':
      priceText = ` with stop $${stopPrice}`;
      break;
    case 'STP_LMT':
      priceText = ` with stop $${stopPrice} limit $${limitPrice}`;
      break;
    case 'TRAIL':
      priceText = ` with trailing stop (${trailAmount ? `$${trailAmount}` : `${trailPercent}%`})`;
      break;
    case 'BRACKET':
      priceText = ` with bracket (TP: $${takeProfitPrice}, SL: $${stopLossPrice})`;
      break;
    case 'OCA':
      priceText = ` in OCA group "${ocaGroup}"`;
      break;
    default:
      priceText = '';
  }

  const attributes = [];
  if (outsideRth) attributes.push('Outside RTH');
  if (hidden) attributes.push('Hidden');
  const attributesText = attributes.length > 0 ? ` (${attributes.join(', ')})` : '';

  return (
    <Text>
      {action} {quantity} shares of {symbol}{priceText}{attributesText}
    </Text>
  );
};

interface IBOrderPlacementProps {
  visible: boolean;
  onClose: () => void;
  onOrderPlaced?: (orderData: any) => void;
}

interface OrderFormData {
  symbol: string;
  action: 'BUY' | 'SELL';
  quantity: number;
  order_type: 'MKT' | 'LMT' | 'STP' | 'STP_LMT' | 'TRAIL' | 'BRACKET' | 'OCA';
  limit_price?: number;
  stop_price?: number;
  trail_amount?: number;
  trail_percent?: number;
  take_profit_price?: number;
  stop_loss_price?: number;
  time_in_force: 'DAY' | 'GTC' | 'IOC' | 'FOK';
  outside_rth?: boolean;
  hidden?: boolean;
  discretionary_amount?: number;
  account_id?: string;
  parent_order_id?: string;
  oca_group?: string;
}

export const IBOrderPlacement: React.FC<IBOrderPlacementProps> = ({
  visible,
  onClose,
  onOrderPlaced
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleSubmit = async (values: OrderFormData) => {
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/ib/orders/place`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(values),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to place order');
      }

      const result = await response.json();
      setSuccess(`Order placed successfully. Order ID: ${result.order_id}`);
      
      if (onOrderPlaced) {
        onOrderPlaced(result);
      }

      // Reset form after successful submission
      form.resetFields();

    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    form.resetFields();
    setError(null);
    setSuccess(null);
    onClose();
  };

  const orderType = Form.useWatch('order_type', form) || 'MKT';

  return (
    <Modal
      title={
        <Space>
          <ShoppingCartOutlined />
          Place IB Order
        </Space>
      }
      open={visible}
      onCancel={handleClose}
      footer={null}
      width={600}
      style={{ top: 20 }}
      bodyStyle={{ 
        maxHeight: '70vh', 
        overflowY: 'auto',
        padding: '16px'
      }}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={{
          action: 'BUY',
          order_type: 'MKT',
          time_in_force: 'DAY',
          quantity: 100
        }}
      >
        {error && (
          <Alert
            message="Order Error"
            description={error}
            type="error"
            style={{ marginBottom: '16px' }}
            showIcon
          />
        )}

        {success && (
          <Alert
            message="Order Success"
            description={success}
            type="success"
            style={{ marginBottom: '16px' }}
            showIcon
          />
        )}

        <Row gutter={12}>
          <Col span={12}>
            <Form.Item
              label="Symbol"
              name="symbol"
              rules={[
                { required: true, message: 'Please enter symbol' },
                { min: 1, max: 20, message: 'Symbol must be 1-20 characters' }
              ]}
            >
              <Input 
                placeholder="e.g. AAPL, MSFT, SPY"
                style={{ textTransform: 'uppercase' }}
                onChange={(e) => {
                  form.setFieldsValue({ symbol: e.target.value.toUpperCase() });
                }}
              />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="Action"
              name="action"
              rules={[{ required: true, message: 'Please select action' }]}
            >
              <Select>
                <Option value="BUY">Buy</Option>
                <Option value="SELL">Sell</Option>
              </Select>
            </Form.Item>
          </Col>
        </Row>

        <Row gutter={12}>
          <Col span={12}>
            <Form.Item
              label="Quantity"
              name="quantity"
              rules={[
                { required: true, message: 'Please enter quantity' },
                { type: 'number', min: 1, message: 'Quantity must be at least 1' }
              ]}
            >
              <InputNumber
                style={{ width: '100%' }}
                placeholder="Number of shares"
                min={1}
                max={999999}
                step={1}
              />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="Order Type"
              name="order_type"
              rules={[{ required: true, message: 'Please select order type' }]}
            >
              <Select>
                <Option value="MKT">Market (MKT)</Option>
                <Option value="LMT">Limit (LMT)</Option>
                <Option value="STP">Stop (STP)</Option>
                <Option value="STP_LMT">Stop Limit (STP_LMT)</Option>
                <Option value="TRAIL">Trailing Stop (TRAIL)</Option>
                <Option value="BRACKET">Bracket Order</Option>
                <Option value="OCA">One-Cancels-All (OCA)</Option>
              </Select>
            </Form.Item>
          </Col>
        </Row>

        {(orderType === 'LMT' || orderType === 'STP_LMT') && (
          <Form.Item
            label={
              <Space>
                <DollarOutlined />
                Limit Price
              </Space>
            }
            name="limit_price"
            rules={[
              { required: true, message: 'Please enter limit price' },
              { type: 'number', min: 0.01, message: 'Price must be greater than 0' }
            ]}
          >
            <InputNumber
              style={{ width: '100%' }}
              placeholder="Limit price"
              min={0.01}
              step={0.01}
              precision={2}
              prefix="$"
            />
          </Form.Item>
        )}

        {(orderType === 'STP' || orderType === 'STP_LMT' || orderType === 'BRACKET') && (
          <Form.Item
            label={
              <Space>
                <DollarOutlined />
                Stop Price
              </Space>
            }
            name="stop_price"
            rules={[
              { required: true, message: 'Please enter stop price' },
              { type: 'number', min: 0.01, message: 'Price must be greater than 0' }
            ]}
          >
            <InputNumber
              style={{ width: '100%' }}
              placeholder="Stop price"
              min={0.01}
              step={0.01}
              precision={2}
              prefix="$"
            />
          </Form.Item>
        )}

        {orderType === 'TRAIL' && (
          <Row gutter={12}>
            <Col span={12}>
              <Form.Item
                label="Trail Amount ($)"
                name="trail_amount"
                rules={[{ type: 'number', min: 0.01, message: 'Amount must be greater than 0' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="Trail amount"
                  min={0.01}
                  step={0.01}
                  precision={2}
                  prefix="$"
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="Trail Percent (%)"
                name="trail_percent"
                rules={[{ type: 'number', min: 0.01, max: 100, message: 'Percent must be between 0.01 and 100' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="Trail percent"
                  min={0.01}
                  max={100}
                  step={0.01}
                  precision={2}
                  suffix="%"
                />
              </Form.Item>
            </Col>
          </Row>
        )}

        {orderType === 'BRACKET' && (
          <Row gutter={12}>
            <Col span={12}>
              <Form.Item
                label="Take Profit Price"
                name="take_profit_price"
                rules={[{ type: 'number', min: 0.01, message: 'Price must be greater than 0' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="Take profit price"
                  min={0.01}
                  step={0.01}
                  precision={2}
                  prefix="$"
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="Stop Loss Price"
                name="stop_loss_price"
                rules={[{ type: 'number', min: 0.01, message: 'Price must be greater than 0' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="Stop loss price"
                  min={0.01}
                  step={0.01}
                  precision={2}
                  prefix="$"
                />
              </Form.Item>
            </Col>
          </Row>
        )}

        {orderType === 'OCA' && (
          <Form.Item
            label="OCA Group"
            name="oca_group"
            rules={[{ required: true, message: 'Please enter OCA group name' }]}
          >
            <Input placeholder="One-Cancels-All group name" />
          </Form.Item>
        )}

        <Row gutter={12}>
          <Col span={12}>
            <Form.Item
              label={
                <Space>
                  <ClockCircleOutlined />
                  Time in Force
                </Space>
              }
              name="time_in_force"
              rules={[{ required: true, message: 'Please select time in force' }]}
            >
              <Select>
                <Option value="DAY">Day (DAY)</Option>
                <Option value="GTC">Good Till Cancelled (GTC)</Option>
                <Option value="IOC">Immediate or Cancel (IOC)</Option>
                <Option value="FOK">Fill or Kill (FOK)</Option>
              </Select>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="Account ID (Optional)"
              name="account_id"
            >
              <Input placeholder="Leave empty for default account" />
            </Form.Item>
          </Col>
        </Row>

        <Row gutter={12}>
          <Col span={8}>
            <Form.Item
              name="outside_rth"
              valuePropName="checked"
            >
              <Checkbox>Allow Outside RTH</Checkbox>
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item
              name="hidden"
              valuePropName="checked"
            >
              <Checkbox>Hidden Order</Checkbox>
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item
              label="Discretionary Amount"
              name="discretionary_amount"
            >
              <InputNumber
                style={{ width: '100%' }}
                placeholder="Discretionary amount"
                min={0}
                step={0.01}
                precision={2}
                prefix="$"
              />
            </Form.Item>
          </Col>
        </Row>

        <Divider />

        <div style={{ marginBottom: '16px' }}>
          <Text type="secondary">
            <strong>Order Summary:</strong>
          </Text>
          <div style={{ marginTop: '8px' }}>
            <OrderSummaryText form={form} orderType={orderType} />
          </div>
        </div>

        <Alert
          message="Risk Warning: Orders sent to Interactive Brokers"
          type="warning"
          style={{ marginBottom: '12px', fontSize: '12px' }}
          showIcon
          banner
        />

        <Form.Item style={{ marginBottom: 0 }}>
          <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
            <Button onClick={handleClose}>
              Cancel
            </Button>
            <Button 
              type="primary" 
              htmlType="submit" 
              loading={loading}
              icon={<ShoppingCartOutlined />}
            >
              Place Order
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default IBOrderPlacement;