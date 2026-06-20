from typing import Any, Optional

def get_param(
    opts: Any, 
    explicit_val: Optional[Any] = None, 
    attr_name: str = None, 
    default_val: Optional[Any] = None
) -> Any:
    """
    Hàm tổng quát để lấy tham số. 
    Thứ tự ưu tiên: Giá trị truyền trực tiếp > Giá trị trong opts > Giá trị mặc định.
    """
    if explicit_val is not None:
        return explicit_val
    
    # Kiểm tra nếu opts tồn tại và có thuộc tính cần tìm
    return getattr(opts, attr_name, default_val) if opts is not None else default_val