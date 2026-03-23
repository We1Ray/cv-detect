"""Tests for shared.user_manager -- role-based user management."""

from __future__ import annotations

import pytest

from shared.user_manager import UserManager, UserRecord, UserRole

# Strong password that meets policy (8+ chars, 1 uppercase, 1 digit, 1 special)
_PW = "TestPass1!"
_PW2 = "NewPass22@"
_PW3 = "AltPass33#"


@pytest.fixture
def manager(tmp_path) -> UserManager:
    """Create a UserManager backed by a temporary database."""
    db = str(tmp_path / "test_users.db")
    return UserManager(db_path=db)


# ------------------------------------------------------------------
# User creation
# ------------------------------------------------------------------


class TestCreateUser:
    """Tests for user creation."""

    def test_create_user(self, manager: UserManager) -> None:
        """A new user should be created and given an integer id."""
        uid = manager.create_user("alice", _PW, UserRole.OPERATOR)
        assert isinstance(uid, int)
        assert uid > 0

    def test_duplicate_username_raises(self, manager: UserManager) -> None:
        """Creating a user with an existing username should raise ValueError."""
        manager.create_user("bob", _PW, UserRole.OPERATOR)
        with pytest.raises(ValueError, match="already exists"):
            manager.create_user("bob", _PW2, UserRole.ENGINEER)

    def test_default_admin_created(self, tmp_path) -> None:
        """On first init a default admin account should exist."""
        db = str(tmp_path / "fresh.db")
        mgr = UserManager(db_path=db)
        # Default admin is created with a random secure password.
        # We verify the admin user exists by listing users.
        users = mgr.list_users()
        admin_users = [u for u in users if u.username == "admin"]
        assert len(admin_users) == 1
        assert admin_users[0].role == UserRole.ADMIN
        assert admin_users[0].force_password_change is True

    def test_weak_password_rejected(self, manager: UserManager) -> None:
        """Passwords that don't meet the policy should be rejected."""
        with pytest.raises(ValueError, match="8 characters"):
            manager.create_user("weak", "short", UserRole.OPERATOR)
        with pytest.raises(ValueError, match="uppercase"):
            manager.create_user("weak", "alllower1", UserRole.OPERATOR)
        with pytest.raises(ValueError, match="digit"):
            manager.create_user("weak", "NoDigitHere", UserRole.OPERATOR)


# ------------------------------------------------------------------
# Authentication
# ------------------------------------------------------------------


class TestAuthentication:
    """Tests for authenticate()."""

    def test_authenticate_success(self, manager: UserManager) -> None:
        """Valid credentials should return a UserRecord."""
        manager.create_user("carol", _PW, UserRole.ENGINEER)
        user = manager.authenticate("carol", _PW)
        assert user is not None
        assert user.username == "carol"
        assert user.role == UserRole.ENGINEER
        assert user.last_login is not None

    def test_authenticate_wrong_password(self, manager: UserManager) -> None:
        """Wrong password should return None."""
        manager.create_user("dave", _PW, UserRole.OPERATOR)
        assert manager.authenticate("dave", "WrongPass9!") is None

    def test_authenticate_nonexistent_user(self, manager: UserManager) -> None:
        """Non-existent user should return None."""
        assert manager.authenticate("ghost", _PW) is None

    def test_authenticate_deactivated_user(self, manager: UserManager) -> None:
        """A deactivated user cannot authenticate."""
        manager.create_user("eve", _PW, UserRole.OPERATOR)
        manager.deactivate_user("eve")
        assert manager.authenticate("eve", _PW) is None


# ------------------------------------------------------------------
# Password management
# ------------------------------------------------------------------


class TestChangePassword:
    """Tests for change_password()."""

    def test_change_password(self, manager: UserManager) -> None:
        """Password change should work with correct old password."""
        manager.create_user("frank", _PW, UserRole.OPERATOR)
        assert manager.change_password("frank", _PW, _PW2) is True
        # Old password no longer works
        assert manager.authenticate("frank", _PW) is None
        # New password works
        assert manager.authenticate("frank", _PW2) is not None

    def test_change_password_wrong_old(self, manager: UserManager) -> None:
        """change_password should return False when old password is wrong."""
        manager.create_user("gina", _PW, UserRole.OPERATOR)
        assert manager.change_password("gina", "FakePass9!", _PW2) is False

    def test_change_to_weak_password_rejected(self, manager: UserManager) -> None:
        """Changing to a weak password should raise ValueError."""
        manager.create_user("hank", _PW, UserRole.OPERATOR)
        with pytest.raises(ValueError):
            manager.change_password("hank", _PW, "weak")


# ------------------------------------------------------------------
# Password policy
# ------------------------------------------------------------------


class TestPasswordPolicy:
    """Tests for check_password_policy."""

    def test_strong_password_accepted(self) -> None:
        ok, msg = UserManager.check_password_policy("StrongP1!")
        assert ok is True
        assert msg == ""

    def test_no_special_char(self) -> None:
        ok, msg = UserManager.check_password_policy("StrongP1")
        assert ok is False
        assert "special" in msg.lower()

    def test_too_short(self) -> None:
        ok, msg = UserManager.check_password_policy("Ab1")
        assert ok is False
        assert "8" in msg

    def test_no_uppercase(self) -> None:
        ok, msg = UserManager.check_password_policy("lowercase1")
        assert ok is False

    def test_no_digit(self) -> None:
        ok, msg = UserManager.check_password_policy("NoDigitHere")
        assert ok is False


# ------------------------------------------------------------------
# Role hierarchy & permissions
# ------------------------------------------------------------------


class TestRoleHierarchy:
    """Tests for check_permission and set_role."""

    def test_role_hierarchy(self, manager: UserManager) -> None:
        """ADMIN >= ENGINEER >= OPERATOR in the permission check."""
        manager.create_user("op", _PW, UserRole.OPERATOR)
        manager.create_user("eng", _PW2, UserRole.ENGINEER)
        manager.create_user("adm", _PW3, UserRole.ADMIN)

        op = manager.authenticate("op", _PW)
        eng = manager.authenticate("eng", _PW2)
        adm = manager.authenticate("adm", _PW3)

        # Operator can access OPERATOR level only
        assert UserManager.check_permission(op, UserRole.OPERATOR) is True
        assert UserManager.check_permission(op, UserRole.ENGINEER) is False
        assert UserManager.check_permission(op, UserRole.ADMIN) is False

        # Engineer can access OPERATOR and ENGINEER
        assert UserManager.check_permission(eng, UserRole.OPERATOR) is True
        assert UserManager.check_permission(eng, UserRole.ENGINEER) is True
        assert UserManager.check_permission(eng, UserRole.ADMIN) is False

        # Admin can access everything
        assert UserManager.check_permission(adm, UserRole.OPERATOR) is True
        assert UserManager.check_permission(adm, UserRole.ENGINEER) is True
        assert UserManager.check_permission(adm, UserRole.ADMIN) is True

    def test_set_role(self, manager: UserManager) -> None:
        """set_role should change the user's role."""
        manager.create_user("hal", _PW, UserRole.OPERATOR)
        manager.set_role("hal", UserRole.ENGINEER)
        users = {u.username: u for u in manager.list_users()}
        assert users["hal"].role == UserRole.ENGINEER

    def test_set_role_nonexistent_raises(self, manager: UserManager) -> None:
        """set_role on a missing user should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            manager.set_role("nobody", UserRole.ADMIN)


# ------------------------------------------------------------------
# User listing & activation
# ------------------------------------------------------------------


class TestListAndActivation:
    """Tests for list_users, deactivate_user, activate_user."""

    def test_list_users(self, manager: UserManager) -> None:
        """list_users should include all users (default admin + created)."""
        manager.create_user("u1", _PW, UserRole.OPERATOR)
        manager.create_user("u2", _PW2, UserRole.ENGINEER)
        users = manager.list_users()
        names = {u.username for u in users}
        assert {"admin", "u1", "u2"}.issubset(names)

    def test_deactivate_user(self, manager: UserManager) -> None:
        """Deactivated user should show is_active=False."""
        manager.create_user("ivan", _PW, UserRole.OPERATOR)
        manager.deactivate_user("ivan")
        users = {u.username: u for u in manager.list_users()}
        assert users["ivan"].is_active is False

    def test_activate_user(self, manager: UserManager) -> None:
        """Re-activating a user should restore login capability."""
        manager.create_user("jane", _PW, UserRole.OPERATOR)
        manager.deactivate_user("jane")
        manager.activate_user("jane")
        assert manager.authenticate("jane", _PW) is not None


# ------------------------------------------------------------------
# Current user session
# ------------------------------------------------------------------


class TestCurrentUser:
    """Tests for get_current_user / set_current_user."""

    def test_current_user_initially_none(self, manager: UserManager) -> None:
        assert manager.get_current_user() is None

    def test_set_and_get_current_user(self, manager: UserManager) -> None:
        manager.create_user("kate", _PW, UserRole.OPERATOR)
        user = manager.authenticate("kate", _PW)
        manager.set_current_user(user)
        assert manager.get_current_user() is not None
        assert manager.get_current_user().username == "kate"
