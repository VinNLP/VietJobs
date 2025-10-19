#!/bin/bash

# Get list of txt files to process
files=(
    links/links_Bán_lẻ_Dịch_vụ_đời_sống.txt
    links/links_Biên_Phiên_dịch.txt
    links/links_Chăm_sóc_khách_hàng_Customer_Service_Vận_hành.txt
    links/links_Công_nghệ_thông_tin.txt
    links/links_Điện_Điện_tử_Viễn_thông.txt
    links/links_Dược_Y_tế_Sức_khỏe_Công_nghệ_sinh_học.txt
    links/links_Giáo_dục_Đào_tạo.txt
    links/links_Kế_toán_Kiểm_toán_Thuế.txt
    links/links_Kinh_doanh_Bán_hàng.txt
    links/links_Lao_động_phổ_thông.txt
    links/links_Logistics_Thu_mua_Kho_vận_tải.txt
    links/links_Luật.txt
    links/links_Marketing_PR_Quảng_cáo.txt
    links/links_Năng_lượng_Môi_trường_Nông_nghiệp.txt
    links/links_Nhà_hàng_Khách_sạn_Du_lịch.txt
    links/links_Nhân_sự_Hành_chính_Pháp_chế.txt
    links/links_Nhóm_nghề_khác.txt
    links/links_Phim_và_Truyền_hình_Báo_chí_Xuất_bản.txt
    links/links_Sản_xuất.txt
    links/links_Tài_chính_Ngân_hàng_Bảo_hiểm.txt
    links/links_Tài_xế.txt
    links/links_Thiết_kế.txt
    links/links_Tư_vấn_Chuyên_môn.txt
    links/links_Xây_dựng_Bất_động_sản.txt
    )

# Process each file
for file in "${files[@]}"; do
    python llm_extract.py "$file"
done

echo "Finished processing all files"
