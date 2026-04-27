"""
rpi_code.py  —  Create WiFi Access Point on Raspberry Pi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Works on Raspberry Pi OS Bookworm (and older) using NetworkManager.
Run ONCE as root — the AP will start automatically on every boot.

Usage:
    sudo python3 rpi_code.py
"""

import subprocess, sys, os

# ══════════════════════════════════════════════════
#  CONFIG — change these if needed
# ══════════════════════════════════════════════════
AP_SSID     = "RobotCar"
AP_PASSWORD = "robot1234"
AP_IFACE    = "wlan0"        # WiFi interface
AP_IP       = "192.168.4.1"  # RPi's IP on the AP network
CON_NAME    = "RobotCar-AP"  # NetworkManager connection name
# ══════════════════════════════════════════════════


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    if check and result.returncode != 0:
        print(f"[ERROR] Command failed (exit {result.returncode}): {cmd}")
        sys.exit(1)
    return result


def nm_available() -> bool:
    return run("which nmcli", check=False).returncode == 0


def setup_ap_nmcli():
    """Create AP using NetworkManager (Raspberry Pi OS Bookworm / modern systems)."""
    print("\n=== Setting up Access Point with NetworkManager ===")

    # Remove old connection with same name if it exists
    run(f"nmcli con delete '{CON_NAME}'", check=False)

    # Create the hotspot connection
    run(f"nmcli con add type wifi ifname {AP_IFACE} con-name '{CON_NAME}' "
        f"autoconnect yes ssid '{AP_SSID}'")

    run(f"nmcli con modify '{CON_NAME}' "
        f"802-11-wireless.mode ap "
        f"802-11-wireless.band bg "
        f"802-11-wireless.channel 6 "
        f"ipv4.method shared "
        f"ipv4.addresses {AP_IP}/24")

    run(f"nmcli con modify '{CON_NAME}' "
        f"wifi-sec.key-mgmt wpa-psk "
        f"wifi-sec.psk '{AP_PASSWORD}'")

    # Bring up the AP
    run(f"nmcli con up '{CON_NAME}'")


def setup_ap_hostapd():
    """Fallback: create AP using hostapd + dnsmasq (older Raspberry Pi OS)."""
    print("\n=== Setting up Access Point with hostapd + dnsmasq ===")

    run("apt-get update -qq")
    run("apt-get install -y hostapd dnsmasq")
    run("systemctl stop hostapd dnsmasq", check=False)

    hostapd_conf = f"""\
interface={AP_IFACE}
driver=nl80211
ssid={AP_SSID}
hw_mode=g
channel=6
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase={AP_PASSWORD}
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
"""
    dnsmasq_conf = f"""\
interface={AP_IFACE}
dhcp-range=192.168.4.2,192.168.4.50,255.255.255.0,24h
"""
    with open("/etc/hostapd/hostapd.conf", "w") as f:
        f.write(hostapd_conf)
    run("sed -i 's|#*DAEMON_CONF=.*|DAEMON_CONF=\"/etc/hostapd/hostapd.conf\"|'"
        " /etc/default/hostapd")

    run("cp /etc/dnsmasq.conf /etc/dnsmasq.conf.orig", check=False)
    with open("/etc/dnsmasq.conf", "w") as f:
        f.write(dnsmasq_conf)

    # Static IP via /etc/network/interfaces (no dhcpcd needed)
    iface_block = f"""
auto {AP_IFACE}
iface {AP_IFACE} inet static
    address {AP_IP}
    netmask 255.255.255.0
"""
    iface_path = "/etc/network/interfaces.d/ap-static"
    with open(iface_path, "w") as f:
        f.write(iface_block)

    run("systemctl unmask hostapd")
    run("systemctl enable hostapd dnsmasq")
    run("systemctl restart hostapd dnsmasq")


def setup_ap():
    if os.geteuid() != 0:
        print("[ERROR] Must run as root:  sudo python3 rpi_code.py")
        sys.exit(1)

    # Enable IP forwarding (needed for internet sharing, harmless otherwise)
    print("\n=== Enabling IPv4 forwarding ===")
    # Write to /etc/sysctl.d/ — works even if /etc/sysctl.conf doesn't exist
    with open("/etc/sysctl.d/99-ip-forward.conf", "w") as f:
        f.write("net.ipv4.ip_forward=1\n")
    run("sysctl -w net.ipv4.ip_forward=1")   # apply immediately without reboot

    if nm_available():
        setup_ap_nmcli()
        active = run("nmcli con show --active", check=False).stdout
        ap_ok  = CON_NAME in active
    else:
        setup_ap_hostapd()
        result = run("systemctl is-active hostapd", check=False)
        ap_ok  = result.stdout.strip() == "active"

    print(f"""
╔══════════════════════════════════════════════╗
║  Access Point {'OK ✓' if ap_ok else 'FAILED ✗ — see errors above'}
║                                              ║
║  SSID     : {AP_SSID:<32} ║
║  Password : {AP_PASSWORD:<32} ║
║  RPi IP   : {AP_IP:<32} ║
╚══════════════════════════════════════════════╝

  1. Connect ESP32 + phone to Wi-Fi: "{AP_SSID}"
  2. Start the car controller:
         python3 rpi_control.py
  3. Open in phone browser:
         http://{AP_IP}:8001
""")

    if not ap_ok:
        print("  [HINT] Check logs:  journalctl -xe -u NetworkManager")
        sys.exit(1)


if __name__ == "__main__":
    setup_ap()
